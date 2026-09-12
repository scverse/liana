from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple
from warnings import warn

import numba as nb
import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import sparray, spmatrix
from scipy.spatial import cKDTree
from scverse_misc import Deprecation
from tqdm import tqdm

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._constants import PrimaryColumns as P
from liana._core._docs import d
from liana._core._pipe_utils import assert_covered, prep_check_adata
from liana._core._pipe_utils._common import _get_groupby_subset
from liana._core._types import MatrixLike, get_obs, get_x
from liana.method.sp._utils import _add_complexes_to_var
from liana.resource.select_resource import _handle_resource

if TYPE_CHECKING:
    prange = range
else:
    prange = nb.prange

_PAIR_CHUNK_DEPRECATION = Deprecation(
    "2.1", "The weighted numerator no longer holds per-chunk temporaries, so there is nothing to tune."
)
"""Why `lric`'s `pair_chunk` no longer does anything."""


type Transform = Callable[[np.ndarray], np.ndarray]
"""Rescales an expression matrix before ligand/receptor weights are formed."""

_MIN_CELLS_FRAC = 0.01

# ── helpers ───────────────────────────────────────────────────────────


def _default_min_cells(adata: AnnData, min_cells: int | None, verbose: bool | None) -> int:
    """Default ``min_cells`` to an abundance-relative threshold.

    ``None`` means "drop cell types making up no more than ``_MIN_CELLS_FRAC``
    of the slide". ``prep_check_adata`` keeps types with ``count >= min_cells``,
    so the threshold is ``floor(frac * N) + 1``: a type sitting exactly at the
    fraction is dropped. An explicit integer passes through untouched.
    """
    if min_cells is not None:
        return min_cells
    default = int(np.floor(_MIN_CELLS_FRAC * adata.n_obs)) + 1
    _logg(
        f"`min_cells=None`: dropping cell types with <= {_MIN_CELLS_FRAC:.1%} of "
        f"{adata.n_obs} cells (min_cells={default}).",
        verbose=verbose,
    )
    return default


def _linear_transform(expr: np.ndarray) -> np.ndarray:
    """Mean-normalise to a mean of 1.

    A column whose mean is 0 is passed through unscaled rather than divided:
    that is an all-zero gene (negative input is rejected upstream by
    ``prep_check_adata(block_negatives=True)``), so the column is already 0 and
    dividing would only turn it into ``NaN``.
    """
    mean = expr.mean(axis=0, keepdims=True)
    smean = np.where(mean > 0, mean, 1.0)
    return expr / smean


def _to_dense(X: MatrixLike) -> np.ndarray:
    dense = X.toarray() if isinstance(X, spmatrix | sparray) else X
    return np.asarray(dense, dtype=np.float32)


def _pair_weights(
    adata: AnnData,
    gene_names: list[str],
    idx: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Per-pair expression weights.

    Transform the unique-gene matrix, then gather to per-pair columns ``idx``;
    returns ``(n_cells, n_pairs)``. ``use_raw``/``layer`` are already resolved
    into ``.X`` by ``prep_check_adata``.
    """
    return transform(_to_dense(get_x(adata[:, gene_names])))[:, idx]


def _index_resource(
    adata: AnnData,
    resource: pd.DataFrame,
    lr_sep: str,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    """
    Filter resource to genes present in ``adata.var_names`` and build integer indices.

    Returns
    -------
    lr_pairs : (n_pairs, 2) int array indexing into ``unique_ligands`` / ``unique_receptors``
    unique_ligands : list of str
    unique_receptors : list of str
    pair_names : list of ``"ligand<lr_sep>receptor"`` labels
    """
    var_set = set(map(str, adata.var_names))
    unique_ligands = [g for g in dict.fromkeys(resource["ligand"]) if g in var_set]
    unique_receptors = [g for g in dict.fromkeys(resource["receptor"]) if g in var_set]
    resource_f = resource[resource["ligand"].isin(unique_ligands) & resource["receptor"].isin(unique_receptors)].copy()
    lig_to_idx = {g: i for i, g in enumerate(unique_ligands)}
    rec_to_idx = {g: i for i, g in enumerate(unique_receptors)}
    lr_pairs = np.column_stack(
        [
            resource_f["ligand"].map(lig_to_idx).to_numpy(),
            resource_f["receptor"].map(rec_to_idx).to_numpy(),
        ]
    ).astype(int)
    pair_names = (resource_f["ligand"].astype(str) + lr_sep + resource_f["receptor"].astype(str)).tolist()
    return lr_pairs, unique_ligands, unique_receptors, pair_names


def _lr_weights(
    adata: AnnData,
    resource: pd.DataFrame,
    lr_sep: str,
    transform_fn: Transform | None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Per-pair ligand/receptor weight matrices, plus one label per LR pair.

    Returns ``(WL, WR, ligands, receptors, pair_names)``, all aligned on the
    LR-pair axis; both weight matrices are ``(n_cells, n_pairs)``.
    """
    transform = _linear_transform if transform_fn is None else transform_fn
    lr_pairs, unique_ligands, unique_receptors, pair_names = _index_resource(adata, resource, lr_sep)
    if lr_pairs.size == 0:
        raise ValueError("No LR pairs found in adata.var_names after filtering the resource.")
    WL = _pair_weights(adata, unique_ligands, lr_pairs[:, 0], transform)
    WR = _pair_weights(adata, unique_receptors, lr_pairs[:, 1], transform)
    ligands = [unique_ligands[i] for i in lr_pairs[:, 0]]
    receptors = [unique_receptors[j] for j in lr_pairs[:, 1]]
    return WL, WR, ligands, receptors, pair_names


def _type_index(adata: AnnData, groupby: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Cell-type labels, their sorted levels, per-cell codes and per-type counts."""
    obs_types = get_obs(adata)[groupby].astype(str).to_numpy()
    levels = sorted(np.unique(obs_types).tolist())
    codes = pd.Categorical(obs_types, categories=levels).codes.astype(np.int64)
    return obs_types, levels, codes, np.bincount(codes, minlength=len(levels))


def _fold_groupby_pairs(
    groupby_pairs: pd.DataFrame | None, cell_types: Sequence[str] | None
) -> NDArray[Any] | Sequence[str] | None:
    """Add the cell types referenced by ``groupby_pairs`` to ``cell_types``."""
    subset = _get_groupby_subset(groupby_pairs)
    if subset is None:
        return cell_types
    return subset if cell_types is None else np.union1d(list(cell_types), subset)


def _check_annulus_steps(annulus_steps: object) -> int:
    """Validate an ``annulus_steps`` argument coming from an untyped caller.

    Takes `object` rather than `int` precisely because the `isinstance` check is the point: a float would otherwise truncate silently.
    """
    if not isinstance(annulus_steps, int | np.integer) or annulus_steps < 1:
        raise ValueError(f"`annulus_steps` must be an integer >= 1, got {annulus_steps!r}.")
    return int(annulus_steps)


def _make_radii(
    max_radius: float, radius_step: float, annulus_steps: int = 1, extend_first_annulus: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Inner/outer edges of the output annuli; each is ``annulus_steps`` radius steps wide."""
    radii_inner = np.arange(radius_step, max_radius + radius_step, radius_step, dtype=float)
    radii_outer = radii_inner + annulus_steps * radius_step
    if extend_first_annulus:
        radii_inner[0] = 0.0  # merge the [0, radius_step) contact band into the first annulus
    return radii_inner, radii_outer


def _fine_tiles(n_bins: int, radius_step: float, annulus_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint ``radius_step``-wide tiles from 0 out to the last annulus' outer edge."""
    fine_inner = np.arange(n_bins + annulus_steps, dtype=float) * radius_step
    return fine_inner, fine_inner + radius_step


def _roll_tiles(fine: np.ndarray, n_bins: int, k: int, extend_first: bool) -> np.ndarray:
    """Sum ``k`` consecutive fine tiles per output annulus."""
    C = np.concatenate([np.zeros((1, *fine.shape[1:])), np.cumsum(fine, axis=0)], axis=0)
    los = np.arange(n_bins) + 1
    his = los + k
    if extend_first:
        los[0] = 0
    return np.asarray(C[his] - C[los])


def _prop_mask(mat: np.ndarray, prop_snd: np.ndarray, prop_rcv: np.ndarray, min_prop: float) -> np.ndarray:
    """NaN out pairs whose sender or receiver expressing-cell proportion is below the threshold."""
    if min_prop <= 0:
        return mat
    filt = (prop_snd < min_prop) | (prop_rcv < min_prop)
    if filt.any():
        mat = mat.copy()
        mat[:, filt] = np.nan
    return mat


@nb.njit(cache=True)
def _bin_edges(
    I: np.ndarray,
    J: np.ndarray,
    D: np.ndarray,
    radii_inner: np.ndarray,
    radii_outer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop self-pairs and pairs outside every tile, and bin what is left.

    One pass over the pair list, counting first and filling second, so that the
    intermediate masks the equivalent chain of numpy expressions would allocate over
    tens of millions of pairs never exist.
    """
    n_bins = radii_inner.size
    n_edges = I.size

    kept = 0
    for e in range(n_edges):
        if I[e] == J[e]:
            continue
        d = np.float64(D[e])
        b = 0
        while b < n_bins and radii_outer[b] <= d:
            b += 1
        if b < n_bins and d >= radii_inner[b]:
            kept += 1

    out_i = np.empty(kept, dtype=I.dtype)
    out_j = np.empty(kept, dtype=J.dtype)
    out_bin = np.empty(kept, dtype=np.int64)

    at = 0
    for e in range(n_edges):
        if I[e] == J[e]:
            continue
        d = np.float64(D[e])
        b = 0
        while b < n_bins and radii_outer[b] <= d:
            b += 1
        if b < n_bins and d >= radii_inner[b]:
            out_i[at] = I[e]
            out_j[at] = J[e]
            out_bin[at] = b
            at += 1

    return out_i, out_j, out_bin


def _support_edge_list(
    tree: cKDTree, radii_inner: np.ndarray, radii_outer: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global (i, j, bin) edge list for all ordered pairs within ``radii_outer[-1]``.

    Self-pairs are excluded, and pairs are binned on the half-open ``[inner, outer)`` convention.
    Each pair is assigned to exactly one bin, so the bins must be disjoint tiles for the counts to be complete.
    """
    spdm = tree.sparse_distance_matrix(tree, max_distance=float(radii_outer[-1]), output_type="coo_matrix")

    return _bin_edges(
        spdm.row,
        spdm.col,
        spdm.data.astype(np.float32),
        np.asarray(radii_inner, dtype=np.float64),
        np.asarray(radii_outer, dtype=np.float64),
    )


class _Support(NamedTuple):
    """The annuli, the single edge list they are read off, and the tissue-wide pair counts."""

    N: int
    radii: np.ndarray  # inner edge of each output annulus
    I: np.ndarray
    J: np.ndarray
    bin_idx: np.ndarray  # fine tile of each edge
    n_fine: int
    T: np.ndarray  # (n_bins,) ordered pairs per annulus, any cell type
    roll: Callable[[np.ndarray], np.ndarray]  # fine tiles -> output annuli


def _build_support(
    adata: AnnData,
    spatial_key: str,
    max_radius: float,
    radius_step: float,
    annulus_steps: int,
    extend_first_annulus: bool,
) -> _Support:
    """Bin every ordered pair once, on disjoint fine tiles."""
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    radii_inner, _ = _make_radii(max_radius, radius_step, annulus_steps, extend_first_annulus)
    n_bins = len(radii_inner)
    fine_inner, fine_outer = _fine_tiles(n_bins, radius_step, annulus_steps)
    n_fine = len(fine_inner)
    I, J, bin_idx = _support_edge_list(cKDTree(coords), fine_inner, fine_outer)
    roll = partial(_roll_tiles, n_bins=n_bins, k=annulus_steps, extend_first=extend_first_annulus)
    T = roll(np.bincount(bin_idx, minlength=n_fine).astype(np.float64))
    return _Support(len(coords), radii_inner, I, J, bin_idx, n_fine, T, roll)


def _expected_pairs(n_S: int, n_R: int, N: int, T: np.ndarray) -> np.ndarray:
    """Ordered sender->receiver pairs expected per annulus under random labelling."""
    return n_S * n_R / (N * (N - 1)) * T


def _group_edges(
    group_key: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort the edge list by ``group_key``, so each group is a contiguous slice.

    The counting pass indexes an array of ``n_groups`` by the key and numba does not
    bounds-check, so a key outside that range would corrupt memory rather than raise.
    One pass over the keys up front is cheap next to the sort it guards.
    """
    if group_key.size and (group_key.min() < 0 or group_key.max() >= n_groups):
        raise ValueError(f"`group_key` must lie in [0, {n_groups}), got [{group_key.min()}, {group_key.max()}].")

    return _counting_sort_edges(group_key, I, J, n_groups)


@nb.njit(cache=True)
def _counting_sort_edges(
    group_key: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place every edge in its group in one pass.

    The key is a cell-type pair crossed with a radius tile, so it spans a few hundred
    values over tens of millions of edges -- a counting sort places every edge in one
    pass, where a comparison sort pays a factor of log(n_edges) for the same order.
    Ascending traversal keeps ties in input order, so the result matches a stable sort
    exactly, and the group offsets fall out of the histogram rather than a second search.
    """
    bounds = np.zeros(n_groups + 1, dtype=np.int64)
    for e in range(group_key.size):
        bounds[group_key[e] + 1] += 1
    for g in range(n_groups):
        bounds[g + 1] += bounds[g]

    I_sorted = np.empty(I.size, dtype=I.dtype)
    J_sorted = np.empty(J.size, dtype=J.dtype)

    cursor = bounds[:n_groups].copy()
    for e in range(group_key.size):
        g = group_key[e]
        at = cursor[g]
        I_sorted[at] = I[e]
        J_sorted[at] = J[e]
        cursor[g] = at + 1

    return I_sorted, J_sorted, bounds


def _selected_levels(levels: list[str], selected_types: Sequence[str] | ArrayLike | None) -> list[str]:
    """The retained cell types the user asked to see, in ``levels`` order.

    The null is always estimated over every retained type (see
    :func:`_select_pairs`), so the selection narrows only what is emitted --
    including the categories of the ``source``/``target`` columns.
    """
    if selected_types is None:
        return levels
    wanted = {str(ct) for ct in np.asarray(selected_types).ravel()}
    return [ct for ct in levels if ct in wanted]


def _select_pairs(
    pairs: list[tuple[int, int]],
    levels: list[str],
    selected_types: Sequence[str] | ArrayLike | None,
    groupby_pairs: pd.DataFrame | None,
    symmetric: bool,
    verbose: bool | None,
    input_types: set[str] | None = None,
) -> list[tuple[int, int]]:
    """Restrict the *emitted* cell-type index ``pairs`` to what the caller asked for.

    ``selected_types`` (``cell_types`` folded together with the types named by
    ``groupby_pairs``) keeps only pairs whose both endpoints were selected;
    ``groupby_pairs`` then keeps only the listed combinations. This is a filter
    on the output alone -- ``levels``, ``counts`` and the point pattern behind
    ``T(r)`` still cover every retained cell type, so the random-labelling null
    is unchanged by the selection.

    ``symmetric`` matches a ``groupby_pairs`` request in either orientation --
    ``cross_pcf``'s ``g(r)`` is orientation-free, whereas ``lric``'s directed
    curves are not.

    ``input_types`` are the cell types of the *unfiltered* input: a requested
    type in it but not in ``levels`` was dropped by ``min_cells`` rather than
    misspelled, and the two need different fixes.
    """
    if selected_types is None and groupby_pairs is None:
        return pairs
    if selected_types is not None:
        wanted = {str(ct) for ct in np.asarray(selected_types).ravel()}
        if absent := wanted.difference(levels):
            dropped = absent.intersection(input_types or ())
            detail = [f"not in the data: {sorted(absent - dropped)}"] if absent - dropped else []
            if dropped:
                detail.append(f"dropped by `min_cells`: {sorted(dropped)}")
            _logg(
                f"`cell_types`/`groupby_pairs` name cell types that are {'; '.join(detail)}.",
                level="warn",
                verbose=verbose,
            )
        pairs = [(s, r) for s, r in pairs if levels[s] in wanted and levels[r] in wanted]
    if groupby_pairs is not None:
        requested = set(zip(groupby_pairs[P.source], groupby_pairs[P.target], strict=True))
        pairs = [
            (s, r)
            for s, r in pairs
            if (levels[s], levels[r]) in requested or (symmetric and (levels[r], levels[s]) in requested)
        ]
    if not pairs:
        _logg(
            "`cell_types`/`groupby_pairs` matched no cell-type pair; the result is empty.",
            level="warn",
            verbose=verbose,
        )
    return pairs


@nb.njit(parallel=True, cache=True)
def _segment_weighted_sums(
    I_sorted: np.ndarray,
    J_sorted: np.ndarray,
    bounds: np.ndarray,
    WL: np.ndarray,
    WR: np.ndarray,
) -> np.ndarray:
    """``out[g, p] = sum_{(i, j) in group g} WL[i, p] * WR[j, p]``, shape ``(n_groups, n_pairs)``.

    ``g`` is a group -- one radius bin for a fixed sender-to-receiver cell-type pair -- and ``p`` a ligand-receptor pair.
    Edges must already be sorted by group key (with ``bounds`` from :func:`_group_edges`) so that each group occupies a contiguous slice.
    Accumulating edge by edge keeps the whole reduction out of temporaries, so nothing has to be tiled to bound memory.
    """
    n_groups = bounds.size - 1
    n_pairs = WL.shape[1]
    out = np.zeros((n_groups, n_pairs), dtype=np.float64)

    for g in prange(n_groups):
        for edge in range(bounds[g], bounds[g + 1]):
            i = I_sorted[edge]
            j = J_sorted[edge]
            for p in range(n_pairs):
                out[g, p] += WL[i, p] * WR[j, p]

    return out


def _melt_curves(
    radii: np.ndarray,
    ids: dict[str, ArrayLike],
    categories: dict[str, Sequence[Any]] | None = None,
    **values: np.ndarray,
) -> pd.DataFrame:
    """Long ("liana_res") frame: one row per curve x radius bin.

    ``ids`` maps column name -> one label per curve; ``values`` maps column name
    -> ``(n_bins, n_curves)`` matrix. Row order is curve-major, bin-minor.
    ``categories`` pins the levels of an id column, keeping cell types that only
    ever appear on one side of a pair.
    """
    n_bins, n_curves = next(iter(values.values())).shape
    df = pd.DataFrame({k: pd.Categorical(np.repeat(v, n_bins)) for k, v in ids.items()})
    for k, levels in (categories or {}).items():
        df[k] = df[k].cat.set_categories(levels)
    df["radius"] = np.tile(np.asarray(radii, dtype=float), n_curves)
    for k, v in values.items():
        df[k] = np.ascontiguousarray(v.T).ravel()
    return df


def _type_mean_weights(W: np.ndarray, obs_types: np.ndarray, cell_types_list: list[str]) -> np.ndarray:
    """Per-type mean weight, shape ``(n_types, n_pairs)``.

    These are the ``w̄L^S`` / ``w̄R^R`` of the conditional null -- the quantity
    that makes a type-level marginal shift cancel exactly. Computed once
    (``O(N * n_pairs)``) and reused by every directed pair involving the type.
    """
    out = np.empty((len(cell_types_list), W.shape[1]), dtype=np.float64)
    for ci, ct in enumerate(cell_types_list):
        out[ci] = W[obs_types == ct].mean(axis=0)
    return out


# ── CrossPCF ───────────────────────────────────────────


class CrossPCF:
    """Cross pair-correlation function (cross-PCF) between cell types.

    The cross-PCF, ``g(r)``, measures whether sender-type and receiver-type cells
    co-localize at distance ``r`` more or less often than expected under
    **random labelling**: the observed cell locations are held fixed while cell-type
    labels are assumed to be randomly assigned to those locations. Equivalently,
    ``g(r)`` compares the observed number of sender→receiver pairs in each distance
    annulus with the number expected if cell identities were independent of
    position on the observed tissue support.

    ``g(r) > 1`` indicates that the two cell types are spatially co-localised at
    distance ``r`` beyond what would be expected from the tissue architecture
    alone, ``g(r) < 1`` indicates spatial avoidance, and ``g(r) ≈ 1`` indicates
    that cell-type labels are spatially independent given the observed point
    pattern.

    Unlike the classical cross-PCF (Bull et al., 2024, doi:10.1101/2024.12.06.627195), which normalises against complete spatial randomness (CSR), this implementation uses a closed-form random-labelling null.
    By conditioning on the observed tissue support, it naturally accounts for
    non-convex tissue boundaries, holes, and fragmentation without estimating an
    effective tissue area.
    """

    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        groupby: str,
        spatial_key: str = K.spatial_key,
        cell_types: Sequence[str] | None = None,
        min_cells: int | None = None,
        max_radius: float = 200,
        radius_step: float = 20,
        annulus_steps: int = 1,
        extend_first_annulus: bool = True,
        groupby_pairs: pd.DataFrame | None = V.groupby_pairs,
        key_added: str = "cross_pcf",
        inplace: bool = V.inplace,
        verbose: bool | None = V.verbose,
    ) -> pd.DataFrame | None:
        """Cross pair-correlation function (cross-PCF) between cell types.

        Computes the distance-resolved cross-PCF ``g(r)`` for every combination
        of cell types present in ``adata.obs[groupby]``, normalised against the
        empirical any-cell-type pair count within this tissue (a
        random-labelling null).

        Parameters
        ----------
        %(adata)s
        %(groupby)s
        %(spatial_key)s
        %(cell_types)s
            Restricts which pairs are *emitted*, not which cells are used:
            ``g(r)`` is normalised against the point pattern of the whole
            slide, so a selection leaves every curve it does emit unchanged.
        %(min_cells)s
            Default ``None`` derives the threshold from slide composition
            instead of using a fixed count: cell types making up no more than
            1% of all cells are dropped.
        %(max_radius)s
        %(radius_step)s
        %(annulus_steps)s
        %(extend_first_annulus)s
        %(groupby_pairs)s
            Restricts the cell-type combinations that are emitted to those
            listed; matched regardless of orientation, as ``g(r)`` is symmetric.
            Like ``cell_types``, it filters the output only.
        %(key_added)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        A long-format ``pandas.DataFrame`` (liana's ``liana_res`` convention)
        with one row per cell-type pair x radius bin and columns
        ``source``, ``target``, ``interaction`` (``"source^target"``),
        ``radius`` (the annulus' inner edge) and ``g``. Returned if
        ``inplace=False``, else ``None`` (stored in ``adata.uns[key_added]``).

        Notes
        -----
        ``g(r)`` is symmetric in ``source``/``target``, so each unordered pair
        is emitted once, with ``source`` before ``target`` in sorted cell-type
        order. Self-pairs are excluded.

        ``CrossPCF`` and ``LRIC`` share the same binning: half-open
        ``[inner, outer)`` tiles read off a single edge list, with distance-0
        pairs between distinct cells counted in the contact band. Pairwise
        ``LRIC``'s ``g_pcf`` therefore equals ``cross_pcf`` exactly.

        The null conditions on the observed point pattern, so it is estimated
        over every retained cell type; ``cell_types`` and ``groupby_pairs``
        select which curves come back, and never move the ones that do.
        ``min_cells`` is the one residual exception: dropping a lowly abundant
        type removes its cells from the support as well, so its neighbours'
        ``g(r)`` shifts slightly. Pass ``min_cells=0`` to keep the full slide.

        Examples
        --------
        >>> import liana as li
        >>> adata = li.ds.generate_toy_spatial()
        >>> adata.obs["cell_type"] = adata.obs["bulk_labels"]
        >>> li.mt.cross_pcf(adata, groupby="cell_type", key_added="cross_pcf")
        >>> list(adata.uns["cross_pcf"].columns)
        ['source', 'target', 'interaction', 'radius', 'g']

        Rank the pairs with :func:`liana.mt.get_lric_auc` and draw one
        with :func:`liana.pl.lric_lineplot`.
        """
        annulus_steps = _check_annulus_steps(annulus_steps)
        selected_types = _fold_groupby_pairs(groupby_pairs, cell_types)
        _adata_orig = adata
        adata = prep_check_adata(
            adata=adata,
            groupby=groupby,
            min_cells=_default_min_cells(adata, min_cells, verbose),
            # deliberately NOT `groupby_subset=selected_types`: the null conditions on the
            # observed point pattern, so subsetting the cells would move `sup.N`/`sup.T` and
            # destroy the very signal it was asked to report. The selection filters the
            # emitted pairs instead (`_select_pairs`).
            use_raw=False,
            layer=None,
            obsm={spatial_key: adata.obsm[spatial_key]},
            complex_sep=None,
            verbose=verbose,
        )

        _, levels, type_code, counts = _type_index(adata, groupby)
        n_types = len(levels)
        # g(r) is symmetric in (sender, receiver) -- emit each unordered pair once
        pairs = [(s, r) for s in range(n_types) for r in range(s + 1, n_types)]
        pairs = _select_pairs(
            pairs,
            levels,
            selected_types,
            groupby_pairs,
            symmetric=True,
            verbose=verbose,
            input_types=set(get_obs(_adata_orig)[groupby].astype(str)),
        )
        _logg(
            f"Computing cross-PCF over {n_types} retained cell types ({len(pairs)} pairs emitted).",
            verbose=verbose,
        )

        # same shared edge list on disjoint fine tiles as LRIC, so numerator and
        # denominator bin every pair identically; a single `bincount` on the
        # composite (sender, receiver, tile) key gives all directed pairs at once
        sup = _build_support(adata, spatial_key, max_radius, radius_step, annulus_steps, extend_first_annulus)
        key = (type_code[sup.I] * n_types + type_code[sup.J]) * sup.n_fine + sup.bin_idx
        O = sup.roll(  # (n_bins, n_types * n_types)
            np.bincount(key, minlength=n_types * n_types * sup.n_fine).reshape(-1, sup.n_fine).T.astype(np.float64)
        )

        G = np.empty((len(sup.radii), len(pairs)), dtype=np.float32)
        for k, (si, ri) in enumerate(pairs):
            with np.errstate(divide="ignore", invalid="ignore"):
                g = O[:, si * n_types + ri] / _expected_pairs(counts[si], counts[ri], sup.N, sup.T)
                g[sup.T == 0] = np.nan
            G[:, k] = g

        emitted = _selected_levels(levels, selected_types)
        res = _melt_curves(
            sup.radii,
            {
                P.source: [levels[s] for s, _ in pairs],
                P.target: [levels[t] for _, t in pairs],
                "interaction": [f"{levels[s]}{V.lr_sep}{levels[t]}" for s, t in pairs],
            },
            # every SELECTED cell type stays a category, incl. any that only appears as
            # `target` -- the unselected ones are part of the null, not of the result
            categories={P.source: emitted, P.target: emitted},
            g=G,
        )
        if inplace:
            _adata_orig.uns[key_added] = res
            return None
        return res


cross_pcf = CrossPCF()

# ── LRIC ───────────────────────────────────────────────


class LRIC:
    """
    Ligand-Receptor Interaction Correlation (LRIC).

    LRIC is an expression-weighted cross pair-correlation function: each cell's
    contribution at distance ``r`` is weighted by its ligand (sender) and
    receptor (receiver) expression. The resulting ``g(r)`` therefore measures
    whether ligand- and receptor-expressing cells are spatially co-enriched at
    distance ``r`` beyond what would be expected under a random-labelling null,
    in which cell locations are held fixed while ligand and receptor weights are
    randomly reassigned among those locations. In pairwise mode that reassignment
    is restricted to within each cell type, so a type's own mean expression level
    is preserved by construction and cannot masquerade as spatial signal.

    ``g(r) > 1`` indicates that ligand- and receptor-expressing cells occur
    together more often than expected from the observed tissue architecture and
    cell-type composition alone, ``g(r) ≈ 1`` indicates that expression provides
    no additional spatial enrichment beyond the underlying tissue structure, and
    ``g(r) < 1`` indicates depletion. This surfaces candidate ligand-receptor
    interactions that are both spatially proximal and co-expressed, a more
    specific signal of potential cell-cell communication than spatial
    co-localisation or co-expression considered alone.

    LRIC builds on the cross pair-correlation function; see
    (:class:`CrossPCF`).
    """

    @d.dedent
    def __call__(
        self,
        adata: AnnData,
        resource: pd.DataFrame | None = V.resource,
        resource_name: str | None = None,
        interactions: list[tuple[str, str]] | None = V.interactions,
        groupby: str | None = None,
        spatial_key: str = K.spatial_key,
        max_radius: float = 200,
        radius_step: float = 20,
        annulus_steps: int = 1,
        extend_first_annulus: bool = True,
        cell_types: Sequence[str] | None = None,
        min_cells: int | None = None,
        groupby_pairs: pd.DataFrame | None = V.groupby_pairs,
        nz_prop: float = V.nz_prop,
        expr_prop: float = V.expr_prop,
        complex_sep: str | None = V.complex_sep,
        lr_sep: str = V.lr_sep,
        transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        use_raw: bool = V.use_raw,
        layer: str | None = V.layer,
        pair_chunk: int | None = None,
        key_added: str = "lric",
        inplace: bool = V.inplace,
        verbose: bool | None = V.verbose,
    ) -> pd.DataFrame | None:
        """

        Ligand-Receptor Interaction Correlation (LRIC).

        Computes an expression-weighted cross-PCF ``g(r)``: each cell's
        contribution at distance ``r`` is weighted by its ligand and receptor
        expression, so ``g(r) > 1`` flags ligand- and receptor-expressing
        cells that are spatially co-enriched beyond cell-type co-localisation
        alone — candidate ligand-receptor interactions that are both proximal
        and co-expressed.

        When ``groupby`` is ``None`` (default), all cells are treated as
        potential senders and receivers (self-pairs excluded), providing a
        global screen for LR pairs with strong spatial co-enrichment signal.

        When ``groupby`` is a column name in ``adata.obs``, the LRIC is
        computed for every directed sender→receiver cell-type pair, and each
        interaction is additionally decomposed into an architecture-only
        (``g_pcf``) and an expression-only (``g_expr``) component.

        Parameters
        ----------
        %(adata)s
        %(resource)s
        %(resource_name)s
        %(interactions)s
        groupby
            Column in ``adata.obs`` used to define cell types. ``None`` runs
            the cell-type-agnostic mode across all cells.
        %(spatial_key)s
        %(max_radius)s
        %(radius_step)s
        %(annulus_steps)s
        %(extend_first_annulus)s
            In LRIC this specifically preserves juxtacrine (direct-contact)
            ligand-receptor signal in the first bin.
        %(cell_types)s
            Only relevant when ``groupby`` is set. Restricts which pairs are
            *emitted*, not which cells are used: the null conditions on the
            point pattern of the whole slide, so a selection leaves every
            curve it does emit unchanged.
        %(min_cells)s
            Default ``None`` derives the threshold from slide composition
            instead of using a fixed count: cell types making up no more than
            1% of all cells are dropped. Unlike ``cell_types``, this one does
            remove cells, so it shifts the null slightly -- pass
            ``min_cells=0`` to keep the full slide.
        %(groupby_pairs)s
            Only relevant when ``groupby`` is set. Restricts the directed
            sender->receiver combinations that are emitted to those listed.
            Like ``cell_types``, it filters the output only.
        %(nz_prop)s
            Only relevant when ``groupby`` is ``None`` (agnostic mode).
            Interactions below the threshold are set to ``NaN``.
        %(expr_prop)s
            Only relevant when ``groupby`` is set; computed within each
            cell type. Interactions below the threshold are set to ``NaN``.
        complex_sep
            Separator used to identify multi-subunit complexes in the resource
            (e.g. ``"_"`` splits ``"ITGAV_ITGB3"`` into its subunits and adds
            the minimum-subunit expression as a new column in ``adata.var``).
            Set to ``None`` to skip complex handling.
        %(lr_sep)s
        transform_fn
            Expression transform applied to ligand and receptor matrices,
            defaulting to mean-normalisation to 1 (``_linear_transform``).
        %(use_raw)s
            Must resolve to non-negative expression; scaled or centred data is
            rejected, as the weights are expression relative to a mean.
        %(layer)s
            Same requirement as ``use_raw``: non-negative expression only.
        pair_chunk
            Deprecated since 2.1 and ignored. The weighted numerator is now
            accumulated without materialising per-chunk temporaries, so there is
            nothing to tune.
        %(key_added)s
        %(inplace)s
        %(verbose)s

        Raises
        ------
        ValueError
            If the resolved expression matrix carries negative values.

        Returns
        -------
        A long-format ``pandas.DataFrame`` (liana's ``liana_res`` convention)
        with one row per interaction x radius bin, returned if
        ``inplace=False``, else ``None`` (stored in ``adata.uns[key_added]``).

        Agnostic mode columns: ``ligand_complex``, ``receptor_complex``,
        ``interaction`` (``"ligand<lr_sep>receptor"``), ``radius`` (the
        annulus' inner edge) and ``g``.

        Pairwise mode additionally carries ``source`` / ``target`` (the sender
        and receiver cell types) and the ``g_expr`` / ``g_pcf`` decomposition,
        where ``g_pcf`` is shared by all LR pairs of a given ``source``
        ->``target``.

        Interactions masked by ``nz_prop`` / ``expr_prop`` are kept as ``NaN`` rows.

        Examples
        --------
        Cell-type-agnostic LRIC across all cells:

        >>> import liana as li
        >>> adata = li.ds.generate_toy_spatial()
        >>> li.mt.lric(adata, resource_name="consensus", key_added="lric")
        >>> list(adata.uns["lric"].columns)
        ['ligand_complex', 'receptor_complex', 'interaction', 'radius', 'g']

        The cell-type pairwise variant additionally decomposes each interaction
        into architecture (``g_pcf``) and expression (``g_expr``) components:

        >>> adata.obs["cell_type"] = adata.obs["bulk_labels"]
        >>> li.mt.lric(adata, resource_name="consensus", groupby="cell_type", key_added="lric_ct")

        Rank the interactions with :func:`liana.mt.get_lric_auc` -- its output
        feeds straight into :func:`liana.pl.dotplot` -- and draw a single
        ``g(r)`` profile with :func:`liana.pl.lric_lineplot`.
        """
        annulus_steps = _check_annulus_steps(annulus_steps)
        resource = _handle_resource(
            interactions=interactions,
            resource=resource,
            resource_name=resource_name,
            x_name="ligand",
            y_name="receptor",
            verbose=verbose,
        )

        selected_types = _fold_groupby_pairs(groupby_pairs, cell_types) if groupby is not None else None

        _adata_orig = adata
        adata = prep_check_adata(
            adata=adata,
            groupby=groupby,
            min_cells=(_default_min_cells(adata, min_cells, verbose) if groupby is not None else None),
            # deliberately NOT `groupby_subset=selected_types`: `_expected_pairs` conditions on
            # the observed point pattern via `sup.N`/`sup.T`, so subsetting the cells would move
            # the null itself. `_select_pairs` filters the emitted curves instead.
            use_raw=use_raw,
            layer=layer,
            obsm={spatial_key: adata.obsm[spatial_key]},
            complex_sep=complex_sep,
            # the weights are expression relative to a per-gene mean, so negative input is
            # rejected; the log1p heuristic stays off, as any positive rescaling cancels
            block_negatives=True,
            check_lognorm=False,
            verbose=verbose,
        )

        if complex_sep is not None:
            entities = np.union1d(resource["ligand"].astype(str), resource["receptor"].astype(str))
            if any(complex_sep in e for e in entities):
                adata = _add_complexes_to_var(adata, entities, complex_sep=complex_sep)

        assert_covered(np.union1d(resource["ligand"], resource["receptor"]), adata.var_names, verbose=verbose)

        if pair_chunk is not None:
            warn(
                f"The argument pair_chunk is deprecated and will be removed in the future. {_PAIR_CHUNK_DEPRECATION}",
                FutureWarning,
                stacklevel=2,
            )

        sup = _build_support(adata, spatial_key, max_radius, radius_step, annulus_steps, extend_first_annulus)
        if groupby is None:
            res = self._agnostic(
                adata=adata,
                resource=resource,
                sup=sup,
                nz_prop=nz_prop,
                lr_sep=lr_sep,
                transform_fn=transform_fn,
                verbose=verbose,
            )
        else:
            res = self._pairwise(
                adata=adata,
                resource=resource,
                groupby=groupby,
                sup=sup,
                selected_types=selected_types,
                # the levels `_pairwise` sees are post-`min_cells`; these are the input's
                input_types=set(get_obs(_adata_orig)[groupby].astype(str)),
                groupby_pairs=groupby_pairs,
                expr_prop=expr_prop,
                lr_sep=lr_sep,
                transform_fn=transform_fn,
                verbose=verbose,
            )

        if inplace:
            _adata_orig.uns[key_added] = res
            return None
        return res

    def _agnostic(
        self,
        adata: AnnData,
        resource: pd.DataFrame,
        sup: _Support,
        nz_prop: float,
        lr_sep: str,
        transform_fn: Transform | None,
        verbose: bool | None,
    ) -> pd.DataFrame:
        """Cell-type-agnostic LRIC across all cells (self-pairs excluded).

        Null: jointly permute which position gets which (ligand, receptor)
        weight *pair*, both weights move together as a unit, since they
        belong to one cell. For a fixed edge (i, j) and random permutation
        pi, ``E_pi[wL(pi(i)) wR(pi(j))] = (S_L*S_R - sum_i wL(i)wR(i)) / (N(N-1))``,
        the average weighted product over all ``N(N-1)`` distinct ordered
        pairs; the self-pair correction term ``sum wL*wR`` accounts for every
        cell being able to pair with itself in the ``S_L*S_R`` product, while
        actual self-pairs are excluded from the edge set. Reduces exactly to
        ``CrossPCF``'s directed curve when weights are one-hot type
        indicators.
        """
        _logg("Running cell-type-agnostic LRIC.", verbose=verbose)

        WL, WR, ligands, receptors, pair_names = _lr_weights(adata, resource, lr_sep, transform_fn)

        # numerator and denominator both come off the SAME edge list on disjoint
        # fine tiles, then get rolled up into the (possibly overlapping) output
        # annuli -- so every pair is counted identically on both sides
        I_sorted, J_sorted, bounds = _group_edges(sup.bin_idx, sup.I, sup.J, sup.n_fine)
        num = sup.roll(_segment_weighted_sums(I_sorted, J_sorted, bounds, WL, WR))

        # closed-form null mean: T(b) * E[wL(i) wR(j)] over random distinct pairs
        S_L, S_R, cross = WL.sum(0), WR.sum(0), (WL * WR).sum(0)
        denom = sup.T[:, None] * ((S_L * S_R - cross) / (sup.N * (sup.N - 1)))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = num / denom
            g[denom == 0] = np.nan
        lric = g.astype(np.float32)

        if nz_prop > 0:
            lric = _prop_mask(lric, (WL > 0).sum(axis=0) / sup.N, (WR > 0).sum(axis=0) / sup.N, nz_prop)
        return _melt_curves(
            sup.radii,
            {
                P.ligand_complex: ligands,
                P.receptor_complex: receptors,
                "interaction": pair_names,
            },
            g=lric,
        )

    def _pairwise(
        self,
        adata: AnnData,
        resource: pd.DataFrame,
        groupby: str,
        sup: _Support,
        selected_types: Sequence[str] | ArrayLike | None,
        input_types: set[str],
        groupby_pairs: pd.DataFrame | None,
        expr_prop: float,
        lr_sep: str,
        transform_fn: Transform | None,
        verbose: bool | None,
    ) -> pd.DataFrame:
        """Cell-type pairwise ("ct") LRIC under the conditional (within-type) null.

        Weights are computed TISSUE-WIDE (all N cells). Null, in two stages:
        (1) roles are assigned across the tissue, giving
        ``E[T_SR(b)] = n_S n_R / (N(N-1)) * T_all(b)``; (2) each cell's
        ``(wL, wR)`` tuple is permuted only among cells of its own type, giving
        ``E[Num_SR(b) | T_SR(b)] = T_SR(b) * w̄L^S * w̄R^R``. Composed::

            E[Num_SR(b)] = n_S n_R / (N(N-1)) * T_all(b) * w̄L^S * w̄R^R

        Because ``w̄L^S`` / ``w̄R^R`` come from the type's own cells, a
        type-level marginal expression shift cancels exactly.

        ``T_SR(b)`` is read straight off the group bounds (edges per group),
        so no per-type KD-tree is needed.

        Reduces exactly to ``CrossPCF`` whenever weights are
        position-independent.

        ``selected_types``/``groupby_pairs`` narrow the emitted curves only:
        ``levels``, ``counts`` and ``sup`` cover every retained cell type, so
        the null stays conditioned on the whole point pattern.
        """
        obs_types, levels, codes, counts = _type_index(adata, groupby)
        n_types = len(levels)
        pairs = [(s, r) for s in range(n_types) for r in range(n_types) if s != r]
        pairs = _select_pairs(
            pairs,
            levels,
            selected_types,
            groupby_pairs,
            symmetric=False,
            verbose=verbose,
            input_types=input_types,
        )
        _logg(
            f"Running LRIC (conditional within-type null) over {n_types} retained cell types "
            f"({len(pairs)} directed pairs emitted).",
            verbose=verbose,
        )

        WL, WR, ligands, receptors, pair_names = _lr_weights(adata, resource, lr_sep, transform_fn)
        n_lr = len(pair_names)

        # per-type marginals: the heart of the conditional null
        mL = _type_mean_weights(WL, obs_types, levels)
        mR = _type_mean_weights(WR, obs_types, levels)

        # everything -- numerator, per-pair and whole-tissue pair counts -- is read
        # off this one edge list on disjoint fine tiles, then rolled up into the
        # (possibly overlapping) output annuli, so both sides of every ratio bin
        # the pairs identically
        n_bins, n_fine = len(sup.radii), sup.n_fine

        # Group the edge list by (sender type, receiver type, tile), so that each
        # directed pair's edges occupy a contiguous slice addressable via `bounds`.
        # On a large slide (big `max_radius` => tens of millions of edges) the edge
        # list dominates memory, so the key is int32 (keys are bounded by
        # n_types^2 * n_fine, far below int32 range) and is built in place.
        type_code = codes.astype(np.int32)
        group_key = type_code[sup.I].astype(np.int32)
        group_key *= n_types
        group_key += type_code[sup.J]
        group_key *= n_fine
        np.add(group_key, sup.bin_idx, out=group_key, casting="unsafe")
        I_sorted, J_sorted, bounds = _group_edges(group_key, sup.I, sup.J, n_types * n_types * n_fine)
        del group_key

        # Per-type expressing proportions -- a mean over a boolean -- computed once
        # (O(N * n_pairs)) and reused by every directed pair involving that type.
        pexp_L = pexp_R = None
        if expr_prop > 0:
            pexp_L = _type_mean_weights(WL > 0, obs_types, levels)
            pexp_R = _type_mean_weights(WR > 0, obs_types, levels)

        # one (n_bins, n_curves) matrix per output column, curves ordered ct-pair-major
        G = np.empty((n_bins, len(pairs) * n_lr), dtype=np.float32)
        E, PC = np.empty_like(G), np.empty_like(G)

        for k, (si, ri) in enumerate(tqdm(pairs, disable=not verbose, desc="LRIC")):
            n_S, n_R = counts[si], counts[ri]

            g0 = (si * n_types + ri) * n_fine
            grp = bounds[g0 : g0 + n_fine + 1]
            Num_SR = sup.roll(_segment_weighted_sums(I_sorted, J_sorted, grp, WL, WR))
            # edges per group == observed ordered S->R pair count per tile
            T_SR = sup.roll(np.diff(grp).astype(np.float64))

            pair_prod = mL[si] * mR[ri]
            exp_T = _expected_pairs(n_S, n_R, sup.N, sup.T)
            expected = exp_T[:, None] * pair_prod[None, :]  # null weighted pair-sum

            with np.errstate(divide="ignore", invalid="ignore"):
                g_full = Num_SR / expected
                g_full[expected == 0] = np.nan
                g_pcf = T_SR / exp_T
                g_pcf[exp_T == 0] = np.nan
                g_expr = Num_SR / (T_SR[:, None] * pair_prod[None, :])
                g_expr[(T_SR[:, None] * pair_prod[None, :]) == 0] = np.nan

            mat = g_full.astype(np.float32)  # architecture x expression coupling
            mat_e = g_expr.astype(np.float32)  # expression coupling ALONE, given where the cells sit
            # set exactly when `expr_prop > 0`
            if pexp_L is not None and pexp_R is not None:
                mat = _prop_mask(mat, pexp_L[si], pexp_R[ri], expr_prop)
                mat_e = _prop_mask(mat_e, pexp_L[si], pexp_R[ri], expr_prop)

            blk = slice(k * n_lr, (k + 1) * n_lr)
            G[:, blk], E[:, blk] = mat, mat_e
            # architecture ALONE (equals cross_pcf) -- shared by every LR pair
            PC[:, blk] = g_pcf[:, None]

        return _melt_curves(
            sup.radii,
            {
                P.source: np.repeat([levels[s] for s, _ in pairs], n_lr),
                P.target: np.repeat([levels[t] for _, t in pairs], n_lr),
                P.ligand_complex: np.tile(ligands, len(pairs)),
                P.receptor_complex: np.tile(receptors, len(pairs)),
                "interaction": np.tile(pair_names, len(pairs)),
            },
            # only the SELECTED types are categories; the rest are null, not result
            categories=dict.fromkeys((P.source, P.target), _selected_levels(levels, selected_types)),
            g=G,
            g_expr=E,
            g_pcf=PC,
        )


lric = LRIC()
