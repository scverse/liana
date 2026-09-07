import pathlib

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from tests._helpers import as_anndata

from liana.method import MistyData
from liana.method.sp._misty._misty_constructs import genericMistyData, lrMistyData
from liana.method.sp._misty._single_view_models import LinearModel, RandomForestModel, RobustLinearModel
from liana.preprocessing import spatial_neighbors


@pytest.fixture
def adata(data_dir: pathlib.Path) -> AnnData:
    """100 cells of the synthetic misty dataset."""
    adata = sc.read_h5ad(data_dir / "synthetic.h5ad")

    return as_anndata(sc.pp.subsample(adata, n_obs=100, copy=True))


def test_misty_para(adata: AnnData) -> None:
    misty = genericMistyData(
        adata,
        bandwidth=10,
        cutoff=0,
        add_juxta=False,
        set_diag=False,
    )
    misty(model=RandomForestModel, bypass_intra=False, seed=42, n_estimators=11)
    assert np.isin(list(misty.uns.keys()), ["target_metrics", "interactions"]).all()
    target_metrics = misty.uns["target_metrics"]
    assert np.sum(target_metrics[["intra", "para"]].to_numpy(), axis=1).sum() == 11.0
    assert target_metrics.shape == (11, 6)

    interactions = misty.uns["interactions"]
    assert interactions.shape == (220, 4)
    assert interactions[interactions["target"] == "ECM"]["importances"].sum().round(8) == 2.0


def test_misty_bypass(adata: AnnData) -> None:
    misty = genericMistyData(adata, bandwidth=10, add_juxta=True, set_diag=True, max_neighs=10, cutoff=0)
    misty(model=RandomForestModel, alphas=1, bypass_intra=True, seed=42, n_estimators=11)
    assert np.isin(["juxta", "para"], misty.uns["target_metrics"].columns).all()
    assert ~np.isin(["intra"], misty.uns["target_metrics"].columns).all()
    assert misty.uns["target_metrics"].shape == (11, 6)
    np.testing.assert_almost_equal(misty.uns["target_metrics"]["multi_R2"].sum(), 0, decimal=5)

    interactions = misty.uns["interactions"]
    assert interactions.shape == (220, 4)
    assert interactions["importances"].sum().round(10) == 22.0
    np.testing.assert_almost_equal(
        interactions[(interactions["target"] == "ligC") & (interactions["predictor"] == "ligA")][
            "importances"
        ].to_numpy(),
        np.array([0.095, 0.07]),
        decimal=3,
    )


def test_misty_groups(adata: AnnData) -> None:
    misty = genericMistyData(
        adata,
        bandwidth=20,
        add_juxta=True,
        set_diag=False,
        cutoff=0,
    )
    misty(
        model=RandomForestModel,
        alphas=1,
        bypass_intra=False,
        seed=42,
        predict_self=True,
        maskby="cell_type",
        n_estimators=11,
    )

    assert misty.uns["target_metrics"].shape == (22, 8)
    perf_actual = misty.uns["target_metrics"].groupby(["intra_group"])["gain_R2"].mean().to_numpy()
    perf_expected = np.array([-0.0124669, -0.0056514])
    np.testing.assert_almost_equal(perf_actual, perf_expected, decimal=2)

    # assert that there are self interactions = var_n * var_n
    interactions = misty.uns["interactions"]
    self_interactions = interactions[(interactions["target"] == interactions["predictor"])]
    # 11 vars * 4 envs * 3 views = 132; NOTE: However, I drop NAs
    assert self_interactions.shape == (44, 5)
    assert self_interactions[self_interactions["view"] == "intra"]["importances"].isna().all()


def test_lr_misty(toy_spatial: AnnData) -> None:
    misty = lrMistyData(toy_spatial, bandwidth=10, set_diag=True, cutoff=0)
    assert misty.shape == (700, 39)

    misty(model=RandomForestModel, n_estimators=10, bypass_intra=True)
    assert misty.uns["target_metrics"].shape == (16, 5)

    interactions = misty.uns["interactions"]
    assert interactions.shape == (367, 4)
    cmplxs = interactions[interactions["target"].str.contains("_")]["target"].unique()
    assert np.isin(["CD8A_CD8B", "CD74_CXCR4"], cmplxs).all()


def test_linear_misty(adata: AnnData) -> None:
    misty = genericMistyData(adata, bandwidth=10, set_diag=False, cutoff=0)
    assert misty.shape == (100, 33)

    misty(model=LinearModel)
    assert misty.uns["target_metrics"].shape == (11, 7)

    assert misty.uns["interactions"].shape == (330, 4)
    actual = misty.uns["interactions"]["importances"].to_numpy().mean()
    np.testing.assert_almost_equal(actual, 0.5135328101662447, decimal=3)


def test_misty_mask(adata: AnnData) -> None:
    misty = genericMistyData(adata, bandwidth=10, set_diag=False, cutoff=0)
    misty = MistyData(misty)
    assert misty.shape == (100, 33)

    misty.mod["intra"].obs["mask"] = misty.mod["intra"].obs == "A"
    misty(model=LinearModel, maskby="mask")

    assert misty.uns["target_metrics"].shape == (11, 7)
    np.testing.assert_almost_equal(misty.uns["target_metrics"]["multi_R2"].mean(), 0.4203699749106394, decimal=3)
    np.testing.assert_almost_equal(misty.uns["target_metrics"]["intra_R2"].mean(), 0.4248588250759459, decimal=3)

    assert misty.uns["interactions"].shape == (330, 4)
    np.testing.assert_almost_equal(misty.uns["interactions"]["importances"].sum(), 149.30560405771703, decimal=0)


def test_misty_uns_preserved_on_reconstruction(adata: AnnData) -> None:
    # https://github.com/scverse/liana/issues/242
    # reconstructing a MistyData from an existing (Misty/Mu)Data must not drop `uns`
    misty = genericMistyData(adata, bandwidth=10, set_diag=False, cutoff=0)
    misty(model=LinearModel)
    assert "target_metrics" in misty.uns
    assert "interactions" in misty.uns

    reconstructed = MistyData(misty)
    assert "target_metrics" in reconstructed.uns
    assert "interactions" in reconstructed.uns
    pd.testing.assert_frame_equal(reconstructed.uns["target_metrics"], misty.uns["target_metrics"])
    pd.testing.assert_frame_equal(reconstructed.uns["interactions"], misty.uns["interactions"])


def test_misty_custom(toy_spatial: AnnData) -> None:
    # keep first 10 vars
    xdata = toy_spatial[:, :10].copy()
    xdata.var.index = "x" + xdata.var.index

    ydata = toy_spatial[:, -10:].copy()
    ydata.var.index = "y" + ydata.var.index
    intra = toy_spatial[:, 25:30].copy()
    misty = MistyData({"intra": intra, "xdata": xdata, "ydata": ydata}, verbose=True)
    misty(model=RobustLinearModel, k_cv=25, seed=420)

    assert misty.uns["interactions"].shape == (120, 4)
    np.testing.assert_almost_equal(misty.uns["interactions"]["importances"].max(), 4.208217548509984, decimal=3)
    np.testing.assert_almost_equal(misty.uns["interactions"]["importances"].min(), -2.905399645316733, decimal=3)
    # the data is random
    np.testing.assert_almost_equal(misty.uns["target_metrics"]["multi_R2"].mean(), 0, decimal=3)


def test_misty_nonaligned(toy_spatial: AnnData) -> None:
    intra = toy_spatial[:, :10].copy()
    intra.var.index = "x" + intra.var.index
    para = toy_spatial[: int(toy_spatial.n_obs * 0.9), -10:].copy()
    para.var.index = "y" + para.var.index
    # Generate connectivities with shape (para.n_obs, intra.n_obs)
    para.obsm["spatial_connectivities"] = np.ones((para.n_obs, intra.n_obs))
    misty = MistyData(
        {"intra": intra, "ydata": para},
        enforce_obs=False,  # NOTE: This is the key parameter
        verbose=True,
    )
    misty(model=LinearModel, k_cv=3)


def test_misty_nonaligned_sparse(toy_spatial: AnnData) -> None:
    """`spatial_neighbors(reference=...)` writes sparse connectivities to .obsm; they must weight fine."""
    intra = toy_spatial[:, :10].copy()
    intra.var.index = "x" + intra.var.index
    para = toy_spatial[: int(toy_spatial.n_obs * 0.9), -10:].copy()
    para.var.index = "y" + para.var.index
    spatial_neighbors(para, bandwidth=200, cutoff=0.1, set_diag=False, reference=np.asarray(intra.obsm["spatial"]))
    assert sparse.issparse(para.obsm["spatial_connectivities"])

    misty = MistyData({"intra": intra, "ydata": para}, enforce_obs=False)
    assert misty.get_weighted_matrix("ydata").shape == (intra.n_obs, para.n_vars)
    misty(model=LinearModel, k_cv=3)
