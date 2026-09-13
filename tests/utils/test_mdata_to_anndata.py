import numpy as np
import pytest
from mudata import MuData
from numpy.testing import assert_almost_equal
from scipy.sparse import csr_matrix
from tests._helpers import as_anndata, get_x

from liana._core._types import MatrixLike
from liana.multisample import mdata_to_anndata
from liana.preprocessing import neg_to_zero, zi_minmax


def test_m_to_adata(toy_mdata: MuData) -> None:
    adata = mdata_to_anndata(
        toy_mdata, x_mod="adata_x", y_mod="adata_y", x_transform=None, y_transform=None, verbose=True
    )
    assert adata.shape == toy_mdata.shape


def test_mdata_transformations(toy_mdata: MuData) -> None:
    # test minmax
    adata = mdata_to_anndata(
        toy_mdata, x_mod="adata_x", y_mod="adata_y", x_transform=zi_minmax, y_transform=zi_minmax, verbose=False
    )
    assert get_x(adata).max() == 1
    assert_almost_equal(get_x(adata).sum(), 1497.3386, decimal=3)

    # test cutoff
    def zi_minmax_cutoff(x: MatrixLike) -> csr_matrix:
        return zi_minmax(x, cutoff=0.25)

    adata = mdata_to_anndata(
        toy_mdata,
        x_mod="adata_x",
        y_mod="adata_y",
        x_transform=zi_minmax_cutoff,
        y_transform=zi_minmax_cutoff,
        verbose=False,
    )
    assert_almost_equal(get_x(adata).sum(), 2120.704, decimal=3)

    # test non-negative
    from scanpy.preprocessing import scale

    scale(toy_mdata.mod["adata_x"])

    adata = mdata_to_anndata(
        toy_mdata, x_mod="adata_x", y_mod="adata_y", x_transform=neg_to_zero, y_transform=None, verbose=False
    )
    assert_almost_equal(get_x(adata).max(), 7.760507, decimal=5)
    assert_almost_equal(get_x(adata).min(), 0, decimal=5)


def test_m_to_adata_non_shared_obs_raises(toy_mdata: MuData) -> None:
    """`an.concat(axis=1)` intersects the obs while `mdata.obs` unions them -- used to be a shape error."""
    x = as_anndata(toy_mdata.mod["adata_x"])
    y = as_anndata(toy_mdata.mod["adata_y"])
    partial = MuData({"adata_x": x.copy(), "adata_y": y[: y.n_obs - 5].copy()})

    with pytest.raises(ValueError, match="do not cover the same observations"):
        mdata_to_anndata(partial, x_mod="adata_x", y_mod="adata_y")


def test_m_to_adata_reversed_obs_order(toy_mdata: MuData) -> None:
    """Equal obs sets in a different order used to label every cell with another cell's metadata."""
    x = as_anndata(toy_mdata.mod["adata_x"])
    y = as_anndata(toy_mdata.mod["adata_y"])
    # `y` first, and reversed, so neither `mdata.obs_names` nor `mdata.obsm` follow `an.concat`'s order
    mdata = MuData({"adata_y": y[::-1].copy(), "adata_x": x.copy()})
    mdata.obs["cell"] = mdata.obs_names
    mdata.obsm["order"] = np.arange(mdata.n_obs).reshape(-1, 1)
    marker = {name: i for i, name in enumerate(mdata.obs_names)}

    adata = mdata_to_anndata(mdata, x_mod="adata_x", y_mod="adata_y")

    assert adata.obs_names.tolist() != mdata.obs_names.tolist()  # the ordering the fix has to survive
    assert adata.obs["cell"].tolist() == adata.obs_names.tolist()
    assert np.asarray(adata.obsm["order"]).ravel().tolist() == [marker[name] for name in adata.obs_names]
