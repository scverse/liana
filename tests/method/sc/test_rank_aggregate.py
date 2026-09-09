import pathlib
from itertools import product
from unittest import TestCase

import pytest
from anndata import AnnData
from mudata import MuData
from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal
from tests._helpers import as_frame

from liana.method import rank_aggregate
from liana.method.sc._liana_pipe import MdataKwargs
from liana.method.sc._rank_aggregate import AggregateClass


def test_consensus_meta() -> None:
    assert isinstance(rank_aggregate, AggregateClass)
    assert rank_aggregate.magnitude == "magnitude_rank"
    assert rank_aggregate.specificity == "specificity_rank"
    assert rank_aggregate.method_name == "Rank_Aggregate"


def test_aggregate_specs() -> None:
    specificity_specs = {
        "CellPhoneDB": ("cellphone_pvals", True),
        "Connectome": ("scaled_weight", False),
        "log2FC": ("lr_logfc", False),
        "NATMI": ("spec_weight", False),
    }
    TestCase().assertDictEqual(rank_aggregate.specificity_specs, specificity_specs)

    magnitude_specs = {
        "CellPhoneDB": ("lr_means", False),
        "Connectome": ("expr_prod", False),
        "NATMI": ("expr_prod", False),
        "SingleCellSignalR": ("lrscore", False),
    }

    TestCase().assertDictEqual(rank_aggregate.magnitude_specs, magnitude_specs)


def test_aggregate_res(toy_adata: AnnData, data_dir: pathlib.Path) -> None:
    lr_res = as_frame(rank_aggregate(toy_adata, groupby="bulk_labels", n_perms=2, seed=1337, inplace=False, n_jobs=1))
    lr_exp = read_csv(data_dir / "aggregate_rank_rest.csv", index_col=0)
    lr_res = lr_res.sort_values(by=list(lr_res.columns))
    lr_exp = lr_exp.sort_values(by=list(lr_res.columns))
    lr_res.index = lr_exp.index
    assert_frame_equal(lr_res, lr_exp, check_dtype=False, check_exact=False, rtol=1e-4, atol=1e-6)


def test_aggregate_all(toy_adata: AnnData) -> None:
    rank_aggregate(
        toy_adata,
        groupby="bulk_labels",
        aggregate_method="mean",
        return_all_lrs=True,
        seed=1,
        n_perms=2,
        n_jobs=1,
        key_added="all_res",
    )
    assert toy_adata.uns["all_res"].shape == (4200, 13)


def test_aggregate_by_sample(toy_adata: AnnData) -> None:
    rank_aggregate.by_sample(
        toy_adata,
        groupby="bulk_labels",
        n_jobs=1,
        n_perms=2,
        return_all_lrs=True,
        sample_key="sample",
        key_added="liana_by_sample",
    )
    lr_by_sample = toy_adata.uns["liana_by_sample"]

    assert "sample" in lr_by_sample.columns
    assert lr_by_sample.shape == (10836, 14)


def test_aggregate_no_perms(toy_adata: AnnData) -> None:
    rank_aggregate(toy_adata, groupby="bulk_labels", return_all_lrs=True, key_added="all_res", n_perms=None)
    res = toy_adata.uns["all_res"]
    assert res.shape == (4200, 12)
    assert res["specificity_rank"].notna().all()  # aggregated over the non-permutation methods only


def test_shared_score_column_ranked_once() -> None:
    """Connectome and NATMI both use `expr_prod`; ranking it twice would invert it."""
    from liana._core._pipe_utils._aggregate import _rank_aggregate

    lr_res = DataFrame({"expr_prod": [3.0, 2.0, 1.0], "lr_means": [3.0, 2.0, 1.0]})
    specs: dict[str, tuple[str, bool | None]] = {
        "Connectome": ("expr_prod", False),
        "NATMI": ("expr_prod", False),
        "CellPhoneDB": ("lr_means", False),
    }
    ranks = _rank_aggregate(lr_res, specs, aggregate_method="mean")
    assert list(ranks) == sorted(ranks)  # the strongest interaction aggregates to the best (lowest) rank
    assert lr_res["expr_prod"].tolist() == [3.0, 2.0, 1.0]  # input left untouched


def test_aggregate_single_method_warns(toy_adata: AnnData, caplog: pytest.LogCaptureFixture) -> None:
    from liana.method.sc._natmi import natmi
    from liana.method.sc._rank_aggregate import _rank_aggregate_meta

    AggregateClass(_rank_aggregate_meta, methods=[natmi])(toy_adata, groupby="bulk_labels", n_perms=None, verbose=True)
    assert "Aggregating over 1 score(s) only: ['spec_weight']" in caplog.text


def test_aggregate_on_mdata(toy_mdata: MuData) -> None:
    toy_mdata.mod["adata_y"].var.index = "scaled:" + toy_mdata.mod["adata_y"].var.index
    interactions = list(product(toy_mdata.mod["adata_x"].var.index, toy_mdata.mod["adata_y"].var.index))
    interactions = interactions[0:10]

    rank_aggregate(
        toy_mdata,
        groupby="bulk_labels",
        n_perms=None,
        mdata_kwargs=MdataKwargs(
            x_mod="adata_x",
            y_mod="adata_y",
            x_transform=None,
            y_transform=None,
        ),
        use_raw=False,
        interactions=interactions,
        verbose=True,
    )

    assert toy_mdata.uns["liana_res"].shape == (144, 12)
