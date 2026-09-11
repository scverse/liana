import numpy as np
import pytest
from anndata import AnnData
from tests._helpers import get_x, not_none

from liana.multisample import estimate_elbow, nmf


@pytest.fixture
def adata(toy_adata: AnnData) -> AnnData:
    """NMF requires non-negative input; the toy `.X` is log-normalised (non-negative)."""
    return toy_adata


def test_run_nmf(adata: AnnData) -> None:
    W, H, _, _ = not_none(nmf(adata, n_components=2, inplace=False))

    assert W.shape == (adata.n_obs, 2)
    assert H.shape == (adata.n_vars, 2)

    nmf(adata, n_components=None, inplace=True, random_state=0, max_iter=20)
    assert "NMF_W" in adata.obsm
    assert "NMF_H" in adata.varm
    assert adata.obsm["NMF_W"].shape == (adata.n_obs, 4)
    assert adata.varm["NMF_H"].shape == (adata.n_vars, 4)
    assert adata.uns["nmf_errors"].shape == (10, 2)
    assert adata.uns["nmf_rank"] == 4


def test_estimate_elbow(adata: AnnData) -> None:
    errors, rank = estimate_elbow(get_x(adata), k_range=range(1, 10), random_state=0, max_iter=20)
    assert rank == 4
    assert errors.shape == (9, 2)
    assert errors["k"].tolist() == list(range(1, 10))
    np.testing.assert_almost_equal(errors["error"].mean(), 0.5086648)


def test_run_nmf_df(adata: AnnData) -> None:
    df = adata.to_df()
    W, H, errors, n_components = not_none(nmf(df=df, n_components=2, random_state=0, max_iter=20))

    assert W.shape == (adata.n_obs, 2)
    assert H.shape == (adata.n_vars, 2)
    assert n_components == 2
    assert errors is None


def test_run_nmf_df_rejects_inplace(adata: AnnData) -> None:
    """`inplace=True` used to be silently discarded on the `df=` route."""
    with pytest.raises(ValueError, match="needs an `AnnData` object"):
        nmf(df=adata.to_df(), n_components=2, inplace=True, random_state=0, max_iter=20)
