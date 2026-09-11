import pathlib
from itertools import product
from typing import Literal
from unittest import TestCase

import pytest
from anndata import AnnData
from mudata import MuData
from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal
from tests._helpers import as_frame

import liana as li
from liana._core._constants import PrimaryColumns as P
from liana.method import aggregate_meta, cellchat, cellphonedb, rank_aggregate
from liana.method import singlecellsignalr as sca
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
    assert toy_adata.uns["all_res"].shape == (4200, 19)


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
    assert lr_by_sample.shape == (10836, 20)


def test_aggregate_no_perms(toy_adata: AnnData) -> None:
    rank_aggregate(toy_adata, groupby="bulk_labels", return_all_lrs=True, key_added="all_res", n_perms=None)
    res = toy_adata.uns["all_res"]
    assert res.shape == (4200, 18)
    assert res["specificity_rank"].notna().all()  # aggregated over the non-permutation methods only


def test_aggregate_keeps_per_entity_columns(toy_adata: AnnData) -> None:
    """The aggregate used to narrow to the four keys and the scores, so `tileplot` could not plot it."""
    lr_res = as_frame(rank_aggregate(toy_adata, groupby="bulk_labels", n_perms=2, seed=1337, inplace=False, n_jobs=1))

    per_entity = ["ligand", "receptor", "ligand_means", "receptor_means", "ligand_props", "receptor_props"]
    assert set(per_entity) <= set(lr_res.columns)
    assert lr_res[per_entity].notna().all().all()
    assert not any(col.endswith("_duplicated") for col in lr_res.columns)
    # the documented tileplot call, which raised on the narrowed frame
    li.pl.tileplot(liana_res=lr_res, fill="means", label="props", top_n=5, orderby="magnitude_rank")


def test_aggregate_supp_columns(toy_adata: AnnData) -> None:
    """`supp_columns` rides along as payload, exactly as it does for a single method."""
    supp = ["ligand_pvals", "receptor_pvals", "ligand_zscores", "mat_mean", "ligand_props"]
    lr_res = as_frame(rank_aggregate(toy_adata, groupby="bulk_labels", n_perms=None, supp_columns=supp, inplace=False))
    default = as_frame(rank_aggregate(toy_adata, groupby="bulk_labels", n_perms=None, inplace=False))

    assert set(supp) <= set(lr_res.columns)
    assert not lr_res.columns.duplicated().any()  # `ligand_props` is kept anyway, asking for it must not duplicate it
    assert lr_res.groupby(P.primary).size().eq(1).all(), "one interaction must stay one row"
    assert set(lr_res.columns) == set(default.columns) | set(supp)  # the default columns are untouched

    with pytest.raises(KeyError, match="not in index"):  # as for the single methods
        rank_aggregate(toy_adata, groupby="bulk_labels", n_perms=None, supp_columns=["bogus_col"], inplace=False)


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


def test_aggregate_single_method_warns(toy_adata: AnnData) -> None:
    from liana.method.sc._natmi import natmi
    from liana.method.sc._rank_aggregate import _rank_aggregate_meta

    with pytest.warns(UserWarning, match=r"Aggregating over 1 score\(s\) only: \['spec_weight'\]"):
        AggregateClass(_rank_aggregate_meta, methods=[natmi])(toy_adata, groupby="bulk_labels", n_perms=None)


def test_aggregate_no_specificity_scores_raises(toy_adata: AnnData) -> None:
    """Both methods score specificity by a permutation p-value, so `n_perms=None` leaves no column."""
    consensus = AggregateClass(aggregate_meta, methods=[cellphonedb, cellchat])

    with pytest.raises(ValueError, match="No score column is available"):
        consensus(toy_adata, groupby="bulk_labels", n_perms=None)

    consensus(toy_adata, groupby="bulk_labels", n_perms=None, consensus_opts=["Magnitude"], inplace=False)


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

    assert toy_mdata.uns["liana_res"].shape == (144, 18)


def test_aggregate_tolerates_methods_disagreeing_on_the_subunit() -> None:
    """Complexes are reassembled per method, so `ligand`/`receptor` must be payload, not join keys.

    `cellchat` reduces a complex by `*_trimean` while every other method uses `*_means`. A subunit
    expressed at 20 in a fifth of the cells and 0 elsewhere has mean 4 but trimean 0, so it is the
    minimum for `cellchat` and the maximum for everyone else. Joining on the subunit would split the
    interaction into two rows, each `NaN` in the other method's score; three methods, because the
    shared payload has to be pruned after *each* merge or pandas refuses the second suffixing.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    genes = ["L", "B1", "B2", "F1", "F2"]

    def group() -> np.ndarray:
        x = np.zeros((n, len(genes)))
        x[:, 0] = rng.uniform(2, 4, n)
        x[rng.choice(n, size=n // 5, replace=False), 1] = 20.0  # B1: dropout-heavy, mean 4, trimean 0
        x[:, 2] = rng.uniform(1.5, 2.5, n)  # B2: mean ~2, trimean ~2
        x[:, 3:] = rng.uniform(1, 2, (n, 2))
        return x

    adata = AnnData(X=np.vstack([group(), group()]).astype(np.float32))
    adata.var_names = genes
    adata.obs_names = [f"c{i}" for i in range(2 * n)]
    adata.obs["ct"] = ["A"] * n + ["B"] * n
    cp_cc = AggregateClass(aggregate_meta, methods=[cellphonedb, cellchat, sca])

    per_method = _aggregate_toy(cp_cc, adata, consensus_opts=False)
    assert isinstance(per_method, dict)
    complex_rows = {name: res[res["receptor_complex"] == "B1_B2"] for name, res in per_method.items()}
    assert set(complex_rows["CellPhoneDB"]["receptor"]) == {"B2"}
    assert set(complex_rows["CellChat"]["receptor"]) == {"B1"}, "the toy no longer makes the methods disagree"

    res = as_frame(_aggregate_toy(cp_cc, adata))
    assert res.groupby(P.primary).size().eq(1).all(), "one interaction must stay one row"
    assert res[["lr_means", "lr_probs", "lrscore", "cellphone_pvals", "cellchat_pvals"]].notna().all().all()
    # the leftmost method in `methods=` decides the subunit
    assert set(res[res["receptor_complex"] == "B1_B2"]["receptor"]) == {"B2"}
    cc_cp = AggregateClass(aggregate_meta, methods=[cellchat, cellphonedb, sca])
    res = as_frame(_aggregate_toy(cc_cp, adata))
    assert set(res[res["receptor_complex"] == "B1_B2"]["receptor"]) == {"B1"}


def _aggregate_toy(
    consensus: AggregateClass, adata: AnnData, consensus_opts: Literal[False] | None = None
) -> DataFrame | dict[str, DataFrame] | None:
    return consensus(
        adata,
        groupby="ct",
        interactions=[("L", "B1_B2"), ("F1", "F2")],
        expr_prop=0.1,
        n_perms=10,
        use_raw=False,
        inplace=False,
        verbose=False,
        consensus_opts=consensus_opts,
    )
