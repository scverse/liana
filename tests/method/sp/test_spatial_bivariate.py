from itertools import product

import numpy as np
import pytest
from anndata import AnnData
from fast_array_utils.conv import to_dense
from mudata import MuData
from tests._helpers import as_anndata, get_layer, get_raw_x, get_x

from liana._core._constants import DefaultValues as V
from liana.datasets._sample_anndata import generate_anndata
from liana.datasets._sample_resource import sample_resource
from liana.method.sp._bivariate._spatial_bivariate import bivariate
from liana.preprocessing.transform import zi_minmax

expected_gmorans = 0.0994394
expected_glee = 0.04854206


@pytest.fixture
def mdata(toy_mdata: MuData) -> MuData:
    """Toy MuData with an additional all-ones connectivity matrix."""
    toy_mdata.obsp["ones"] = np.ones((toy_mdata.shape[0], toy_mdata.shape[0]), dtype=np.float64)

    return toy_mdata


@pytest.fixture
def interactions(mdata: MuData) -> list[tuple[str, str]]:
    """All possible x-y variable combinations."""
    return list(product(mdata.mod["adata_x"].var.index, mdata.mod["adata_y"].var.index))


def test_bivar_morans_perms(mdata: MuData, interactions: list[tuple[str, str]]) -> None:
    lrdata = as_anndata(
        bivariate(
            mdata,
            x_mod="adata_x",
            y_mod="adata_y",
            local_name="morans",
            n_perms=2,
            nz_prop=0,
            x_use_raw=False,
            y_use_raw=False,
            interactions=interactions,
        )
    )

    local_pvals = get_layer(lrdata, "pvals")
    np.testing.assert_almost_equal(get_x(lrdata).sum(), -346.55872, decimal=2)
    np.testing.assert_almost_equal(np.mean(local_pvals), 0.52787581, decimal=4)


def test_bivar_nondefault(mdata: MuData, interactions: list[tuple[str, str]]) -> None:
    lrdata = as_anndata(
        bivariate(
            mdata,
            x_mod="adata_x",
            y_mod="adata_y",
            local_name="morans",
            global_name=["morans", "lee"],
            n_perms=0,
            nz_prop=0,
            connectivity_key="ones",
            remove_self_interactions=False,
            x_layer="scaled",
            y_layer="scaled",
            x_use_raw=False,
            y_use_raw=False,
            add_categories=True,
            interactions=interactions,
        )
    )

    global_stats = lrdata.var
    np.testing.assert_almost_equal(global_stats["morans"].sum(), 0)
    np.testing.assert_almost_equal(global_stats["lee"].sum(), 0)
    assert global_stats["lee_pvals"].unique()[0] is None
    assert "morans_pvals" in global_stats.columns

    assert lrdata.shape == (680, 100)
    # a "ones" connectivity leaves the local statistic at ~0, so the two-sided
    # analytical p-value is ~1; it was ~0.5 while the p-values were one-sided
    np.testing.assert_almost_equal(np.min(np.min(get_layer(lrdata, "pvals"))), 1.0, decimal=2)


def test_masked_spearman(mdata: MuData, interactions: list[tuple[str, str]]) -> None:
    lrdata = as_anndata(
        bivariate(
            mdata,
            x_mod="adata_x",
            y_mod="adata_y",
            x_use_raw=False,
            y_use_raw=False,
            nz_prop=0,
            local_name="masked_spearman",
            interactions=interactions,
            connectivity_key="ones",
        )
    )
    np.testing.assert_almost_equal(get_x(lrdata).mean(), 0.0027796142, decimal=5)

    assert lrdata.var.shape == (90, 8)
    global_res = lrdata.var
    assert {"mean", "std"}.issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res["mean"].mean(), 0.0027795848, decimal=5)
    # under the `ones` connectivity every neighbourhood is the whole slide, so every spot scores the
    # same; a spread would mean the score depends on cell order again (`std` itself is float32 noise)
    local_scores = to_dense(get_x(lrdata))
    assert (local_scores == local_scores[0]).all()


def test_vectorized_spearman(mdata: MuData, interactions: list[tuple[str, str]]) -> None:
    bdata = as_anndata(
        bivariate(
            mdata,
            x_mod="adata_x",
            y_mod="adata_y",
            x_use_raw=False,
            y_use_raw=False,
            local_name="spearman",
            nz_prop=0,
            n_perms=2,
            interactions=interactions,
        )
    )
    np.testing.assert_almost_equal(get_x(bdata).mean(), 0.0023014963, decimal=5)
    np.testing.assert_almost_equal(get_layer(bdata, "pvals").mean(), 0.7204575163, decimal=3)

    global_res = bdata.var
    assert {"mean", "std"}.issubset(global_res.columns)
    np.testing.assert_almost_equal(global_res["mean"].mean(), 0.0023014963, decimal=5)
    np.testing.assert_almost_equal(global_res["std"].mean(), 0.32339879, decimal=5)


### Test on AnnData and LRs
# NOTE: these should be the same regardless of the local function
def test_morans_analytical(toy_spatial: AnnData) -> None:
    annotations = (set(toy_spatial.obsm), set(toy_spatial.uns), set(toy_spatial.obsp))

    lrdata = as_anndata(
        bivariate(
            toy_spatial,
            local_name="morans",
            global_name=["morans"],
            resource_name=V.resource_name,
            n_perms=0,
            use_raw=True,
            mask_negatives=True,
        )
    )
    assert "pvals" in lrdata.layers.keys()

    # the caller's object is read from, never stripped
    assert (set(toy_spatial.obsm), set(toy_spatial.uns), set(toy_spatial.obsp)) == annotations

    np.testing.assert_almost_equal(np.mean(to_dense(get_x(lrdata[:, "MIF^CD74_CXCR4"]))), 0.12803833, decimal=6)
    # value shifted when the spurious constant was removed from the local Moran's R null variance
    np.testing.assert_almost_equal(np.mean(get_layer(lrdata[:, "MIF^CD74_CXCR4"], "pvals")), 0.8672064, decimal=6)

    interaction = lrdata.var[lrdata.var.index == "S100A9^ITGB2"]
    np.testing.assert_almost_equal(interaction["morans"].to_numpy(), expected_gmorans)
    np.testing.assert_almost_equal(interaction["morans_pvals"].to_numpy(), 3.4125671e-07)


def test_cosine_permutation(toy_spatial: AnnData) -> None:
    toy_spatial.layers["array"] = to_dense(get_raw_x(toy_spatial))
    lrdata = as_anndata(
        bivariate(
            toy_spatial,
            local_name="cosine",
            global_name=["morans", "lee"],
            resource_name="consensus",
            n_perms=100,
            use_raw=False,
            layer="array",
        )
    )

    assert "pvals" in lrdata.layers.keys()

    np.testing.assert_almost_equal(get_x(lrdata[:, "MIF^CD74_CXCR4"]).mean(), 0.32514292, decimal=6)
    np.testing.assert_almost_equal(np.mean(get_layer(lrdata[:, "MIF^CD74_CXCR4"], "pvals")), 0.601228, decimal=4)

    interaction = lrdata.var[lrdata.var.index == "S100A9^ITGB2"]
    np.testing.assert_almost_equal(interaction["mean"].to_numpy(), 0.56016606)
    np.testing.assert_almost_equal(interaction["std"].to_numpy(), 0.33243373)
    np.testing.assert_almost_equal(interaction["morans"].to_numpy(), expected_gmorans)
    np.testing.assert_almost_equal(interaction["morans_pvals"].to_numpy(), 0.85)
    np.testing.assert_almost_equal(interaction["lee"].to_numpy(), expected_glee)
    np.testing.assert_almost_equal(interaction["lee_pvals"].to_numpy(), 0.93)


def test_jaccard_pval_none_cats(toy_spatial: AnnData) -> None:
    lrdata = as_anndata(
        bivariate(
            toy_spatial,
            local_name="jaccard",
            global_name="lee",
            resource_name="consensus",
            n_perms=None,
            use_raw=True,
            add_categories=True,
        )
    )
    assert lrdata.var.shape == (32, 10)

    assert "cats" in lrdata.layers.keys()
    assert get_layer(lrdata, "cats").sum() == -6197
    # `n_perms=None` skips permutations, so no p-values are computed
    assert "pvals" not in lrdata.layers.keys()
    assert "morans_pvals" not in lrdata.var.columns
    interaction = lrdata.var[lrdata.var.index == "S100A9^ITGB2"]
    np.testing.assert_almost_equal(interaction["lee"].to_numpy(), expected_glee)

    np.testing.assert_almost_equal(get_x(lrdata[:, "S100A9^ITGB2"]).mean(), 0.4117572, decimal=6)


def test_bivar_product(mdata: MuData, interactions: list[tuple[str, str]]) -> None:
    from scipy.sparse import csr_matrix

    conn = mdata.obsp["spatial_connectivities"]
    mdata.obsp["norm"] = csr_matrix(conn / conn.sum(axis=1))
    bdata = as_anndata(
        bivariate(
            mdata,
            x_mod="adata_x",
            y_mod="adata_y",
            x_transform=zi_minmax,
            y_transform=zi_minmax,
            x_use_raw=False,
            y_use_raw=False,
            connectivity_key="norm",
            local_name="product",
            global_name=None,
            interactions=interactions,
            n_perms=None,
            add_categories=True,
        )
    )
    assert "cats" in bdata.layers.keys()
    # the cell-level annotations are carried over onto the interaction-level object
    assert bdata.uns.keys() == mdata.uns.keys()
    assert bdata.obsm.keys() == mdata.obsm.keys()
    assert bdata.obsp.keys() == mdata.obsp.keys()
    np.testing.assert_array_equal(bdata.obsm["spatial"], mdata.obsm["spatial"])
    np.testing.assert_almost_equal(get_x(bdata).max(), 0.63145)
    assert "lee" not in bdata.var.columns


def test_large_adata() -> None:
    adata = generate_anndata(n_obs=10001)
    resource = sample_resource(adata, n_lrs=20)
    lrdata = as_anndata(
        bivariate(
            adata,
            resource=resource,
            local_name="pearson",
            global_name="morans",
            n_perms=None,
            use_raw=False,
            add_categories=False,
        )
    )
    np.testing.assert_almost_equal(get_x(lrdata).mean(), 0.00048977, decimal=4)
    np.testing.assert_almost_equal(lrdata.var["morans"].mean(), 0.00012773558, decimal=4)


def test_wrong_interactions(toy_spatial: AnnData) -> None:
    from pytest import raises

    with raises(ValueError):
        bivariate(
            toy_spatial,
            resource_name="mouseconsensus",
            local_name="morans",
            n_perms=None,
            use_raw=True,
            add_categories=True,
        )


def test_wrong_kwargs(toy_spatial: AnnData) -> None:
    from pytest import raises

    with raises(ValueError):
        bivariate(
            toy_spatial,
            resource_name="mouseconsensus",
            local_name="morans",
            n_perms=None,
            use_raw=True,
            add_categories=True,
            life="is good",
        )


def test_show_bivariate() -> None:
    local_scores = bivariate.show_functions()
    assert local_scores.shape == (8, 3)
