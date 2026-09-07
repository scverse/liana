import numpy as np
import pandas as pd
from anndata import AnnData

from liana._core._types import get_coordinates, get_obs


def expand_coordinates(
    adata: AnnData,
    sample_key: str,
    spatial_key: str = "spatial",
    n_cols: int | None = None,
    margin: float = 0.1,
) -> AnnData:
    """
    Lay out the spatial coordinates of multiple samples side-by-side on a grid.

    Each sample is translated into its own cell of a regular grid so that samples
    no longer overlap in coordinate space (e.g. for joint plotting). Within a
    sample the transformation is a pure translation, so relative distances are
    preserved. The original coordinates are stored in ``obsm[f'{spatial_key}_original']``.

    Parameters
    ----------
    adata
        Annotated data object with spatial coordinates in ``obsm[spatial_key]``.
    sample_key
        Column in ``adata.obs`` identifying the sample each observation belongs to.
    spatial_key
        Key in ``adata.obsm`` holding the (n_obs, 2) coordinate matrix.
    n_cols
        Number of columns in the grid. Defaults to a roughly square layout.
    margin
        Fractional spacing between grid cells, relative to the largest sample
        extent. ``margin=0`` packs cells edge-to-edge.

    Returns
    -------
    A copy of ``adata`` with expanded coordinates in ``obsm[spatial_key]`` and the
    original coordinates preserved in ``obsm[f'{spatial_key}_original']``.

    Examples
    --------
    Lays samples that all live in the same coordinate range out on a grid, so that
    they no longer overlap when plotted or when neighbours are computed:

    >>> import numpy as np
    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> rng = np.random.default_rng(0)
    >>> adata.obs["sample"] = rng.choice(["A", "B", "C", "D"], size=adata.n_obs)
    >>> expanded = li.pp.expand_coordinates(adata, sample_key="sample")
    """
    if sample_key not in get_obs(adata):
        raise ValueError(f"`sample_key` '{sample_key}' was not found in `adata.obs`.")
    if spatial_key not in adata.obsm:
        raise ValueError(f"`spatial_key` '{spatial_key}' was not found in `adata.obsm`.")

    adata = adata.copy()
    original = get_coordinates(adata, spatial_key)

    samples = get_obs(adata)[sample_key]
    if isinstance(samples.dtype, pd.CategoricalDtype):
        categories = list(samples.cat.categories)
    else:
        categories = list(pd.unique(samples))
    n_samples = len(categories)

    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_cols = max(1, n_cols)

    samples_arr = samples.to_numpy()

    # per-sample local coordinates (shifted so each sample starts at the origin)
    local = original.copy()
    extents = np.zeros((n_samples, 2))
    for i, sample in enumerate(categories):
        mask = samples_arr == sample
        sample_coords = original[mask]
        mins = sample_coords.min(axis=0)
        local[mask] = sample_coords - mins
        extents[i] = sample_coords.max(axis=0) - mins

    # cell size = largest sample extent (per axis) inflated by the margin
    cell_w, cell_h = extents.max(axis=0) * (1.0 + margin)

    expanded = local.copy()
    for i, sample in enumerate(categories):
        mask = samples_arr == sample
        col = i % n_cols
        row = i // n_cols
        expanded[mask] += np.array([col * cell_w, row * cell_h])

    adata.obsm[f"{spatial_key}_original"] = original
    adata.obsm[spatial_key] = expanded

    return adata
