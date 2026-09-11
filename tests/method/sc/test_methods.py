from itertools import product

import numpy as np
import pandas
import pytest
from anndata import AnnData
from mudata import MuData
from numpy import isclose, max, min
from numpy.testing import assert_almost_equal
from pandas import Categorical, DataFrame

from liana.method import cellchat, cellphonedb, connectome, geometric_mean, logfc, natmi, scseqcomm
from liana.method import singlecellsignalr as sca
from liana.method.sc._liana_pipe import MdataKwargs


@pytest.fixture
def expected_shape(toy_adata: AnnData) -> tuple[int, int]:
    """The methods write to `.uns`, so `.shape` must be left untouched."""
    return toy_adata.shape


def test_cellchat(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    cellchat(toy_adata, groupby="bulk_labels", n_perms=4)

    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "lr_probs" in liana_res.columns
    assert "cellchat_pvals" in liana_res.columns
    assert isclose(max(liana_res[(liana_res.ligand == "TIMP1")].lr_probs), 0.20561589810421071)
    assert isclose(liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["lr_probs"].max(), 0.10198416583005679)
    assert isclose(liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["cellchat_pvals"].mean(), 0.5125)


def test_cellphonedb(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    # legacy check: use_raw=True (read .raw) still works; siblings use the .X default
    cellphonedb(toy_adata, groupby="bulk_labels", use_raw=True, n_perms=4)

    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "lr_means" in liana_res.columns
    assert "cellphone_pvals" in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lr_means), 2.134743630886078, decimal=6)
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["lr_means"].max(), 1.4035000205039978, decimal=6
    )
    assert liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["cellphone_pvals"].mean() == 0.415


def test_cellphonedb_none(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    cellphonedb(toy_adata, groupby="bulk_labels", n_perms=None)
    assert toy_adata.shape == expected_shape
    liana_res = toy_adata.uns["liana_res"]
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["lr_means"].max(), 1.4035000205039978, decimal=6
    )
    assert "cellphone_pvals" not in liana_res.columns


def test_geometric_mean(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    geometric_mean(toy_adata, groupby="bulk_labels", n_perms=4, n_jobs=2)

    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "lr_gmeans" in liana_res.columns
    assert "gmean_pvals" in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lr_gmeans), 2.126363309240465, decimal=6)
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["lr_gmeans"].max(), 1.4016519940029961, decimal=6
    )
    assert liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["gmean_pvals"].mean() == 0.5125


def test_natmi(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    # legacy check: use_raw=True (read .raw) still works; siblings use the .X default
    natmi(toy_adata, groupby="bulk_labels", use_raw=True)

    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "expr_prod" in liana_res.columns
    assert "spec_weight" in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].spec_weight), 0.0604750001773605, decimal=6)
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].expr_prod), 4.521420922884062, decimal=6)
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["spec_weight"].max(), 0.03480120361979308, decimal=6
    )
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["expr_prod"].max(), 1.9646283122925752, decimal=6
    )


def test_scseqcomm(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    scseqcomm(toy_adata, groupby="bulk_labels", expr_prop=0, return_all_lrs=True)

    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "inter_score" in liana_res.columns

    assert_almost_equal(
        liana_res[
            (liana_res.ligand == "TIMP1")
            & (liana_res.receptor == "CD63")
            & (liana_res.source == "Dendritic")
            & (liana_res.target == "CD4+/CD45RA+/CD25- Naive T")
        ]["inter_score"].to_numpy(),
        0.6819619345,
        decimal=5,
    )
    # scored by CXCR4, the limiting subunit of CD74_CXCR4 -- picking the highly expressed
    # CD74 instead saturated `inter_score` at exactly 1.0
    assert_almost_equal(
        max(liana_res[(liana_res.receptor_complex == "CD74_CXCR4")]["inter_score"]), 0.999443104603042, decimal=6
    )


def test_sca(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    sca(toy_adata, groupby="bulk_labels")

    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "lrscore" in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lrscore), 0.781133536891427, decimal=6)
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["lrscore"].max(), 0.7017243729003677, decimal=6
    )


def test_logfc(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    logfc(toy_adata, groupby="bulk_labels")

    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "lr_logfc" in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].lr_logfc), 1.4352725744247437, decimal=6)
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["lr_logfc"].max(), 1.0422011613845825, decimal=6
    )


def test_connectome(toy_adata: AnnData, expected_shape: tuple[int, int]) -> None:
    connectome(toy_adata, groupby="bulk_labels")
    assert toy_adata.shape == expected_shape
    assert "liana_res" in toy_adata.uns.keys()

    liana_res = toy_adata.uns["liana_res"]
    assert isinstance(liana_res, pandas.DataFrame)

    assert "expr_prod" in liana_res.columns
    assert "scaled_weight" in liana_res.columns
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].scaled_weight), 0.9669451713562012, decimal=6)
    assert_almost_equal(max(liana_res[(liana_res.ligand == "TIMP1")].expr_prod), 4.521420922884062, decimal=6)
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["scaled_weight"].max(), 0.9002860486507416, decimal=6
    )
    assert_almost_equal(
        liana_res[liana_res["receptor_complex"] == "CD74_CXCR4"]["expr_prod"].max(), 1.9646283122925752, decimal=6
    )


def test_with_all_lrs(toy_adata: AnnData) -> None:
    natmi(toy_adata, groupby="bulk_labels", return_all_lrs=True, key_added="all_res")
    lr_all = toy_adata.uns["all_res"]
    assert lr_all.shape == (4200, 15)
    assert all(lr_all[~lr_all.lrs_to_keep][natmi.magnitude] == min(lr_all[natmi.magnitude])) is True
    assert all(lr_all[~lr_all.lrs_to_keep][natmi.specificity] == min(lr_all[natmi.specificity])) is True


def test_methods_by_sample(toy_adata: AnnData) -> None:
    logfc.by_sample(toy_adata, groupby="bulk_labels", return_all_lrs=True, sample_key="sample")
    lr_by_sample = toy_adata.uns["liana_res"]

    assert "sample" in lr_by_sample.columns
    assert lr_by_sample.shape == (10836, 15)


def test_methods_by_sample_not_inplace(toy_adata: AnnData) -> None:
    lr_by_sample = logfc.by_sample(toy_adata, groupby="bulk_labels", sample_key="sample", inplace=False)

    assert lr_by_sample is not None
    assert "sample" in lr_by_sample.columns
    assert "liana_res" not in toy_adata.uns


def test_methods_on_mdata(toy_mdata: MuData) -> None:
    toy_mdata.mod["adata_y"].var.index = "scaled:" + toy_mdata.mod["adata_y"].var.index
    interactions = list(product(toy_mdata.mod["adata_x"].var.index, toy_mdata.mod["adata_y"].var.index))
    interactions = interactions[0:10]

    sca(
        toy_mdata,
        groupby="bulk_labels",
        n_perms=None,
        use_raw=False,
        interactions=interactions,
        verbose=True,
        mdata_kwargs=MdataKwargs(
            x_mod="adata_x",
            y_mod="adata_y",
            x_transform=None,
            y_transform=None,
        ),
    )

    assert toy_mdata.uns["liana_res"].shape == (144, 12)


def test_methods_by_sample_on_mdata(toy_mdata: MuData) -> None:
    """`by_sample` slices its input, and slicing a MuData yields a MuData, not an AnnData."""
    toy_mdata.mod["adata_y"].var.index = "scaled:" + toy_mdata.mod["adata_y"].var.index
    interactions = list(product(toy_mdata.mod["adata_x"].var.index, toy_mdata.mod["adata_y"].var.index))[0:10]
    # two balanced samples, so that every cell-type group survives `min_cells` in both
    toy_mdata.obs["sample"] = Categorical(np.where(np.arange(toy_mdata.shape[0]) % 2 == 0, "a", "b"))

    lr_by_sample = sca.by_sample(
        toy_mdata,
        sample_key="sample",
        groupby="bulk_labels",
        n_perms=None,
        use_raw=False,
        interactions=interactions,
        inplace=False,
        mdata_kwargs=MdataKwargs(x_mod="adata_x", y_mod="adata_y", x_transform=None, y_transform=None),
    )

    assert lr_by_sample is not None
    assert sorted(lr_by_sample["sample"].unique()) == ["a", "b"]
    assert {"ligand_complex", "receptor_complex", "lrscore"} <= set(lr_by_sample.columns)


def test_expr_prop_empties_frame_raises(toy_adata: AnnData) -> None:
    """An `expr_prop` that filters out every pair used to fail obscurely, or silently return 0 rows."""
    with pytest.raises(ValueError, match=r"No ligand-receptor pair passed `expr_prop=1.0`"):
        natmi(toy_adata, groupby="bulk_labels", expr_prop=1.0, n_perms=None)


def test_wrong_resource(toy_adata: AnnData) -> None:
    from pytest import raises

    with raises(ValueError):
        natmi(toy_adata, resource_name="mouseconsensus", groupby="bulk_labels", n_perms=4)

    with raises(ValueError):
        natmi(toy_adata, interactions=[("x", "D")], groupby="bulk_labels", n_perms=4)

    with raises(ValueError):
        resource = DataFrame({"ligand": ["A", "B"], "receptor": ["C", "D"]})
        natmi(toy_adata, resource=resource, groupby="bulk_labels", n_perms=4)


def test_spatial_key_opt_in(toy_spatial: AnnData) -> None:
    unweighted = cellphonedb(toy_spatial, groupby="bulk_labels", n_perms=None, use_raw=True, inplace=False)
    weighted = cellphonedb(
        toy_spatial, groupby="bulk_labels", n_perms=None, use_raw=True, inplace=False, spatial_key="spatial"
    )
    assert unweighted is not None and weighted is not None
    assert "proximity" not in unweighted.columns
    assert not isclose(unweighted["lr_means"].sum(), weighted["lr_means"].sum())
    with pytest.raises(KeyError, match="not found in `adata.obsm`"):
        cellphonedb(toy_spatial, groupby="bulk_labels", n_perms=None, use_raw=True, spatial_key="missing")


def test_spatial_key_weights_pvals(toy_spatial: AnnData) -> None:
    """Proximity has to reach the p-values, not just the magnitudes.

    Weighting the permuted null by the same factor as the observed score cancels out of
    `perm * w >= obs * w`, which left every p-value untouched.
    """
    keys = ["source", "target", "ligand_complex", "receptor_complex"]

    unweighted = cellphonedb(toy_spatial, groupby="bulk_labels", n_perms=50, use_raw=True, inplace=False)
    weighted = cellphonedb(
        toy_spatial, groupby="bulk_labels", n_perms=50, use_raw=True, inplace=False, spatial_key="spatial"
    )
    assert unweighted is not None and weighted is not None

    merged = unweighted.merge(weighted, on=keys, suffixes=("_unweighted", "_weighted"))

    assert not merged["cellphone_pvals_unweighted"].equals(merged["cellphone_pvals_weighted"])
    assert (merged["cellphone_pvals_weighted"] >= merged["cellphone_pvals_unweighted"]).all()
