from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scanpy.datasets import pbmc68k_reduced

from liana._core._types import get_obs, get_raw_x, get_x
from liana.preprocessing.spatial_neighbors import spatial_neighbors

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData


def generate_toy_spatial() -> AnnData:
    """
    Build a toy spatial AnnData from `pbmc68k_reduced`.

    Coordinates are drawn at random (seeded), so they carry no biological signal
    -- this is for exercising code paths, not for interpreting results.

    Returns
    -------
    `pbmc68k_reduced` with random `obsm['spatial']` coordinates and the spatial
    connectivities that :func:`liana.pp.spatial_neighbors` derives from them.
    """
    adata = pbmc68k_reduced()
    adata.X = get_raw_x(adata).copy()  # log-norm expression in .X (scverse convention); .raw kept

    rng = np.random.default_rng(seed=1337)
    x = rng.integers(low=0, high=5000, size=adata.shape[0])
    y = rng.integers(low=0, high=5000, size=adata.shape[0])
    adata.obsm["spatial"] = np.array([x, y]).T
    spatial_neighbors(adata, bandwidth=100, cutoff=0.1, set_diag=True)

    return adata


def generate_toy_mdata() -> MuData:
    """
    Build a toy two-view MuData from `pbmc68k_reduced`.

    The two views (`'adata_x'` and `'adata_y'`) hold the same 10 genes, so any
    relationship learnt between them is trivial -- this is for exercising code
    paths, not for interpreting results. Spatial coordinates, connectivities and
    `.obs` are carried over from :func:`liana.ds.generate_toy_spatial`.

    Returns
    -------
    A MuData object with two views, `'adata_x'` and `'adata_y'`, each with a
    `'scaled'` layer, and with `.obsm['spatial']`,
    `.obsp['spatial_connectivities']`, `.obs` and `.uns` shared at the top level.
    """
    import scanpy as sc
    from mudata import MuData

    adata = generate_toy_spatial()
    if adata.raw is None:
        raise ValueError("`adata.raw` is not initialized.")
    adata = adata.raw.to_adata()
    adata = adata[:, 0:10]
    sc.pp.filter_cells(adata, min_counts=1)

    adata.layers["scaled"] = sc.pp.scale(get_x(adata), zero_center=True, max_value=5)

    adata_y = adata.copy()

    # create mdata
    mdata = MuData({"adata_x": adata, "adata_y": adata_y})
    mdata.obsp = adata.obsp
    mdata.uns = adata.uns
    mdata.obsm = adata.obsm
    mdata.obs = get_obs(adata)

    return mdata


def generate_toy_adata() -> AnnData:
    """
    Build a toy multi-sample AnnData from `pbmc68k_reduced`.

    Returns
    -------
    `pbmc68k_reduced` with a randomly-assigned (seeded) `obs['sample']` of four
    samples, and an `obs['case']` splitting those samples into two conditions.
    """
    adata = pbmc68k_reduced()
    adata.X = get_raw_x(adata).copy()  # log-norm expression in .X (scverse convention); .raw kept
    sample_key = "sample"

    rng = np.random.default_rng(0)

    # create fake samples
    obs = get_obs(adata)
    obs[sample_key] = rng.choice(["A", "B", "C", "D"], size=len(obs))

    # group samples into conditions
    obs["case"] = obs[sample_key].map({"A": "yes", "B": "yes", "C": "no", "D": "no"})

    return adata


def generate_anndata(
    sparsity: float = 0.90,
    n_ct: int = 10,
    n_vars: int = 2000,
    n_obs: int = 1000,
    seed: int = 1337,
) -> AnnData:
    # TODO, eventually change completely to use this function, inplace of the other ones
    import scanpy as sc
    from scipy.sparse import csr_matrix

    from liana.preprocessing import spatial_neighbors

    rng = np.random.default_rng(seed=seed)
    dense_counts = rng.poisson(100, size=(n_obs, n_vars))
    mask = rng.choice([0, 1], size=(n_obs, n_vars), p=[sparsity, 1 - sparsity])
    counts = csr_matrix(dense_counts * mask, dtype=np.float32)

    adata = sc.AnnData(counts)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.var_names = [f"Gene{i:d}" for i in range(adata.n_vars)]
    adata.obs_names = [f"Cell{i:d}" for i in range(adata.n_obs)]
    X = get_x(adata)
    nnz = int(np.count_nonzero(X)) if isinstance(X, np.ndarray) else X.nnz
    print(f" NNZ fraction: {nnz / (X.shape[0] * X.shape[1])}")

    x = rng.integers(low=0, high=5000, size=adata.shape[0])
    y = rng.integers(low=0, high=5000, size=adata.shape[0])
    adata.obsm["spatial"] = np.array([x, y]).T

    spatial_neighbors(adata, cutoff=0.1, bandwidth=150, max_neighbours=10)

    # assign cell types
    ct = rng.choice([f"CT{i:d}" for i in range(n_ct)], size=(adata.n_obs,))
    ct = rng.choice(ct, size=(adata.n_obs,))
    get_obs(adata)["cell_type"] = pd.Categorical(ct)

    return adata
