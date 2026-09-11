from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from numpy.typing import ArrayLike, NDArray
from scipy.stats import trim_mean
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._pipe_utils._pre import _require_groupby
from liana._core._types import get_coordinates, get_obs

type _Kernel = Literal["gaussian", "exponential", "linear", "misty_rbf"]
type _Bandwidth = float | np.floating


def _gaussian(distance_mtx: NDArray[np.floating], bandwidth: _Bandwidth) -> NDArray[np.floating]:
    return np.exp(-(distance_mtx**2.0) / (2.0 * bandwidth**2.0))


def _misty_rbf(distance_mtx: NDArray[np.floating], bandwidth: _Bandwidth) -> NDArray[np.floating]:
    return np.exp(-(distance_mtx**2.0) / (bandwidth**2.0))


def _exponential(distance_mtx: NDArray[np.floating], bandwidth: _Bandwidth) -> NDArray[np.floating]:
    return np.exp(-distance_mtx / bandwidth)


def _linear(distance_mtx: NDArray[np.floating], bandwidth: _Bandwidth) -> NDArray[np.floating]:
    connectivity: NDArray[np.floating] = 1 - distance_mtx / bandwidth
    # `np.clip(..., a_min=0, a_max=inf)`, spelled so the result stays typed.
    connectivity[connectivity < 0] = 0
    return connectivity


def _kernel_function(
    distance_mtx: NDArray[np.floating],
    bandwidth: _Bandwidth,
    kernel: _Kernel,
) -> NDArray[np.floating]:
    families = ["gaussian", "exponential", "linear", "misty_rbf"]
    if kernel not in families:
        raise ValueError(f"`kernel` must be one of {families}, got {kernel!r}.")

    if kernel == "gaussian":
        return _gaussian(distance_mtx, bandwidth)
    if kernel == "misty_rbf":
        return _misty_rbf(distance_mtx, bandwidth)
    if kernel == "exponential":
        return _exponential(distance_mtx, bandwidth)
    return _linear(distance_mtx, bandwidth)


def _kernel_scalar(distance: _Bandwidth, bandwidth: _Bandwidth, kernel: _Kernel) -> float:
    """:func:`_kernel_function` applied to a single aggregated distance."""
    return float(_kernel_function(np.atleast_1d(np.float64(distance)), bandwidth, kernel)[0])


@d.dedent
def spatial_neighbors(
    adata: AnnData | MuData,
    bandwidth: float | None = None,
    cutoff: float | None = 0.1,
    max_neighbours: int = 100,
    kernel: _Kernel = "gaussian",
    set_diag: bool = False,
    zoi: float = 0,
    standardize: bool = False,
    reference: ArrayLike | None = None,
    spatial_key: str = K.spatial_key,
    key_added: str = K.spatial_key,
    inplace: bool = V.inplace,
    verbose: bool | None = V.verbose,
) -> np.ndarray | None:
    """
    Generate spatial connectivity weights using Euclidean distance.

    Parameters
    ----------
    adata
        Annotated data object (`AnnData` or `MuData`) with spatial coordinates in `.obsm[spatial_key]`.
    %(bandwidth)s
    cutoff
        Values below this cutoff will be set to 0.
    max_neighbours
        Maximum nearest neighbours to be considered when generating spatial connectivity weights.
        Essentially, the maximum number of edges in the spatial connectivity graph.
        Note that whenever more than `max_neighbours` spots fall within `cutoff` this truncates
        the neighbourhood, and the resulting connectivities are no longer symmetric.
        Clamped down to the number of available locations (`reference` when given, else the spots
        themselves), since fewer neighbours than that cannot be found; logged under `verbose=True`.
    %(kernel)s
    set_diag
        Logical, sets connectivity diagonal to 0 if `False`. Default is `False`.
    zoi
        Zone of indifference. Values below this cutoff will be set to `np.inf`.
    standardize
        Whether to (l1) standardize spatial proximities (connectivities) so that
        they sum to 1. This plays a role when weighing border regions prior to
        downstream methods, as the number of spots in the border region (and
        hence the sum of proximities) is smaller than the number of spots in the
        center. Relevant for methods with unstandardized scores (e.g. product).
        Default is `False`.
    reference
        Reference coordinates to use when generating spatial connectivity
        weights. If `None`, uses the spatial coordinates in
        `adata.obsm[spatial_key]`. This is only relevant if you want to use a
        different set of coordinates to generate spatial connectivity weights.
    %(spatial_key)s
    key_added
        Key to add to `adata.obsp` if `inplace = True`. If reference is not
        `None`, key will be added to `adata.obsm` and any existing `.obsp` entry
        under the same key is left untouched -- methods that read `.obsp` would
        then keep using that older graph, so a warning is emitted.
    %(inplace)s
    %(verbose)s

    Notes
    -----
    This function is adapted from mistyR, and is set to be consistent with
    the `squidpy.gr.spatial_neighbors` function in the `squidpy` package.

    Returns
    -------
    If ``inplace = False``, returns an `np.array` with spatial connectivity
    weights. Otherwise, modifies the ``adata`` object with the following key:

        - :attr:`anndata.AnnData.obsp` ``['{key_added}_connectivities']`` with
          the aforementioned array

    Raises
    ------
    ValueError
        If no ``cutoff`` or ``bandwidth`` are provided
    AssertionError
        If the provided ``spatial_key`` is not in ``adata.obs`` or if ``kernel``
        function is not valid.

    Examples
    --------
    Weights every pair of spots by how close they are, giving the spatially
    informed methods the neighbourhood they operate over:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> li.pp.spatial_neighbors(adata, bandwidth=500)

    `bandwidth` is required and sets the distance over which proximity decays --
    :func:`liana.pp.query_bandwidth` helps pick it -- while `kernel` sets the shape
    of that decay.
    """
    if cutoff is None:
        raise ValueError("`cutoff` must be provided!")
    families = ["gaussian", "exponential", "linear", "misty_rbf"]
    if kernel not in families:
        raise ValueError(f"`kernel` must be one of {families}, got {kernel!r}.")
    if bandwidth is None:
        raise ValueError("Please specify a bandwidth")

    coordinates = get_coordinates(adata, spatial_key)

    _reference = np.asarray(coordinates if reference is None else reference)

    # `+1` to exclude self, but `NearestNeighbors` rejects `n_neighbors > n_samples_fit`, and the tree is
    # fit on `_reference` -- which is `coordinates` only when no `reference` was given.
    n_neighbors = min(max_neighbours + 1, _reference.shape[0])
    if n_neighbors < max_neighbours + 1:
        # info, not warn: with the default `max_neighbours=100` this fires on every
        # dataset (or `reference`) of at most 100 locations, where the cap changes nothing
        _logg(
            f"`max_neighbours={max_neighbours}` needs {max_neighbours + 1} of the "
            f"{_reference.shape[0]} reference location(s) (self included), so it is capped at "
            f"{n_neighbors - 1} neighbour(s).",
            level="info",
            verbose=verbose,
        )

    tree = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="ball_tree",
        metric="euclidean",
    ).fit(_reference)
    dist = tree.kneighbors_graph(coordinates, mode="distance")

    # prevent float overflow
    bandwidth_f = np.float64(bandwidth)

    # define zone of indifference
    dist.data[dist.data < zoi] = np.inf

    # NOTE: dist gets converted to a connectivity (proximity) matrix
    dist.data = _kernel_function(dist.data, bandwidth_f, kernel)

    if not set_diag:
        dist.setdiag(0)
    if cutoff is not None:
        dist.data = dist.data * (dist.data > cutoff)
    if standardize:
        dist = normalize(dist, axis=1, norm="l1")

    spot_n = dist.shape[0]
    if reference is None:
        if spot_n != adata.shape[0]:
            raise RuntimeError(f"built {spot_n} rows of connectivities for {adata.shape[0]} observations.")
    if spot_n > 1000:
        dist = dist.astype(np.float32)

    if inplace:
        if reference is not None:
            # A `reference` graph is not square in `adata`'s observations, so it goes to `.obsm` and
            # `.obsp` is left as-is. Methods that read connectivities from `.obsp` raise only when the
            # key is *absent*, so an entry left over from an earlier call would be reused as if it
            # matched this one.
            if f"{key_added}_connectivities" in adata.obsp:
                _logg(
                    f"`adata.obsp['{key_added}_connectivities']` already exists and is left unchanged: "
                    f"a `reference` graph is written to `.obsm` instead. Methods reading `.obsp` will "
                    f"keep using the older graph. Pass a different `key_added` to keep the two apart.",
                    level="warn",
                )
            adata.obsm[f"{key_added}_connectivities"] = dist
        else:
            adata.obsp[f"{key_added}_connectivities"] = dist

    return None if inplace else dist


@d.dedent
def spatial_pair_proximity(
    adata: AnnData,
    groupby: str,
    spatial_key: str = "spatial",
    bandwidth: float = 250,
    contact_bandwidth: float | None = None,
    min_cells_in_proximity: int = 10,
    trim_fraction: float = 0.1,
    kernel: _Kernel = "gaussian",
    verbose: bool | None = V.verbose,
) -> pd.DataFrame:
    """
    Computes aggregated spatial statistics and proximity scores between cell types.

    This function calculates pairwise proximity between cell types based on nearest neighbor
    distances in spatial coordinates. It returns a DataFrame with proximity scores that can
    be used to weight ligand-receptor interactions by spatial co-localization.

    Parameters
    ----------
    %(adata)s
    %(groupby)s
    %(spatial_key)s
    %(bandwidth)s
    %(contact_bandwidth)s
    min_cells_in_proximity : int, optional
        Minimum number of cell pairs within range required to flag an interaction as significant.
        Default is 10.
    trim_fraction : float, optional
        Fraction of outliers to trim from each tail when calculating mean distance (0-0.5).
        Default is 0.1 (trim 10% from each tail).
    %(kernel)s
    %(verbose)s

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - source: source cell type
        - target: target cell type
        - mean_distance: trimmed mean distance between cell types
        - interacting: binary flag (1 if >= min_cells_in_proximity pairs within bandwidth, else 0)
        - proximity: proximity score calculated by applying kernel to mean_distance with bandwidth
        - contact_interacting: (optional, if contact_bandwidth is not None) binary flag for contact interactions
        - contact_proximity: (optional, if contact_bandwidth is not None) proximity score using contact_bandwidth

    Notes
    -----
    - Performance scales as O(n_cell_types² × n_cells), which is acceptable for typical datasets
      (5-30 cell types) but may be slower with 100+ cell types.
    - Self-interactions exclude the cell itself as its own neighbor to avoid zero distances.
    - Missing proximity values (e.g., cell types that never co-localize) will result in NaN,
      which should be filled with 0.0 when merging with interaction results.

    Examples
    --------
    Summarises how close each pair of `groupby` labels sits in space, which can then
    be used to down-weight interactions between cell types that never meet:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> proximity = li.pp.spatial_pair_proximity(adata, groupby="bulk_labels")

    One row per ordered pair of groups, scored from 1 for groups sitting on top of
    each other down towards 0 for groups that never meet:

    >>> proximity[["source", "target", "proximity"]].head(3).round(3)
               source          target  proximity
    0  CD14+ Monocyte  CD14+ Monocyte      0.663
    1  CD14+ Monocyte         CD19+ B      0.628
    2  CD14+ Monocyte           CD34+      0.020
    """
    _require_groupby(adata, groupby)

    # groupby_labels use categories if categorical
    groupby_labels = np.asarray(get_obs(adata)[groupby])
    coordinates = get_coordinates(adata, spatial_key)

    unique_types = np.unique(groupby_labels)
    stats_list = []

    # Iterate through all cell type pairs
    pair_iterator = [(type_a, type_b) for type_a in unique_types for type_b in unique_types]

    for type_a, type_b in tqdm(pair_iterator, desc="Computing cell type proximities", disable=not verbose):
        idx_a = np.where(groupby_labels == type_a)[0]
        coords_a = coordinates[idx_a]
        idx_b = np.where(groupby_labels == type_b)[0]
        coords_b = coordinates[idx_b]

        if len(idx_a) == 0 or len(idx_b) == 0:
            continue

        # Handle self-interaction (exclude cell itself as neighbor)
        is_self = type_a == type_b
        k_neighbors = 2 if is_self else 1

        if is_self and len(idx_b) < 2:
            continue

        # Nearest neighbor search (1-NN)
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean", n_jobs=-1)
        nn.fit(coords_b)
        distances, _ = nn.kneighbors(coords_a)

        # If self, take 2nd column; if different, take 1st column
        raw_dists = distances[:, 1] if is_self else distances[:, 0]

        # --- Aggregation ---

        # 1. Trimmed mean distance (core metric)
        avg_dist = trim_mean(raw_dists, proportiontocut=trim_fraction)

        # 2. Binary flags (significance based on counts)
        count_long = np.sum(raw_dists <= bandwidth)
        is_interacting = count_long >= min_cells_in_proximity

        # 3. Proximity score (kernel applied to mean_distance)
        prox_score = _kernel_scalar(avg_dist, bandwidth=bandwidth, kernel=kernel)

        # Build result dict
        result_dict: dict[str, str | float | int] = {
            "source": type_a,
            "target": type_b,
            "mean_distance": avg_dist,
            "interacting": int(is_interacting),
            "proximity": prox_score,
        }

        # 4. Optional contact proximity
        if contact_bandwidth is not None:
            count_short = np.sum(raw_dists <= contact_bandwidth)
            is_physically_interacting = count_short >= min_cells_in_proximity
            contact_prox_score = _kernel_scalar(avg_dist, bandwidth=contact_bandwidth, kernel=kernel)

            result_dict["contact_interacting"] = int(is_physically_interacting)
            result_dict["contact_proximity"] = contact_prox_score

        stats_list.append(result_dict)

    return pd.DataFrame(stats_list)
