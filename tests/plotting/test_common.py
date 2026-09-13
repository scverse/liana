import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from pandas import DataFrame

from liana._core._common import _get_liana_res
from liana.plotting._common import (
    _aggregate_scores,
    _check_var,
    _filter_by,
    _filter_labels,
    _get_top_n,
    _invert_scores,
    _prep_liana_res,
)


def test_check_var(liana_res: DataFrame) -> None:
    with pytest.raises(ValueError, match="`size` must be provided"):
        _check_var(liana_res, var_name="size", var=None)

    with pytest.raises(ValueError, match="`not_a_column` \\(size\\) must be one of"):
        _check_var(liana_res, var_name="size", var="not_a_column")

    # a column that exists is returned, so callers can bind a non-optional name
    assert _check_var(liana_res, var_name="size", var="magnitude") == "magnitude"


def test_get_liana_res(toy_adata: AnnData, liana_res: DataFrame) -> None:
    toy_adata.uns["liana_res"] = liana_res

    # an explicitly passed frame takes precedence over `adata.uns[uns_key]`, since `adata` is
    # required by some plots and would otherwise leave no way to plot a frame of one's own
    subset = liana_res.head(3)
    both = _get_liana_res(toy_adata, liana_res=subset)
    assert both.equals(subset)
    assert both is not subset

    # ... and `adata` is read when no frame is given, also as a copy
    from_adata = _get_liana_res(toy_adata, liana_res=None)
    assert from_adata.equals(liana_res)
    assert from_adata is not liana_res

    from_res = _get_liana_res(adata=None, liana_res=liana_res)
    assert from_res.equals(liana_res)
    assert from_res is not liana_res

    with pytest.raises(ValueError, match="`liana_res` or AnnData with `uns_key`"):
        _get_liana_res(adata=None, liana_res=None)


def test_prep_liana_res_filters_complexes(liana_res: DataFrame) -> None:
    ligand = liana_res["ligand_complex"].iloc[0]
    receptor = liana_res["receptor_complex"].iloc[0]

    prepped = _prep_liana_res(liana_res=liana_res, ligand_complex=ligand, receptor_complex=receptor)
    assert set(prepped["ligand_complex"]) == {ligand}
    assert set(prepped["receptor_complex"]) == {receptor}
    assert (prepped["interaction"] == f"{ligand} -> {receptor}").all()


def test_filter_labels(liana_res: DataFrame) -> None:
    # a single label is accepted as a string
    assert set(_filter_labels(liana_res, labels="A", label_type="source")["source"]) == {"A"}

    with pytest.raises(ValueError, match=r"\['Z'\] not found in `liana_res\['source'\]`"):
        _filter_labels(liana_res, labels=["A", "Z"], label_type="source")

    # nothing to filter by
    assert _filter_labels(liana_res, labels=None, label_type="source") is liana_res


def test_aggregate_scores(liana_res: DataFrame) -> None:
    liana_res["interaction"] = liana_res["ligand_complex"] + " -> " + liana_res["receptor_complex"]

    aggregated = _aggregate_scores(
        liana_res.copy(), what="magnitude", how="max", absolute=False, entities=["interaction"]
    )
    assert set(aggregated.columns) == {"interaction", "score"}
    assert aggregated["interaction"].nunique() == aggregated.shape[0]

    absolute = _aggregate_scores(liana_res.copy(), what="magnitude", how="max", absolute=True, entities=["interaction"])
    assert (absolute["score"] >= 0).all()


def test_invert_scores() -> None:
    np.testing.assert_almost_equal(_invert_scores(np.array([1.0])), 0)
    # smaller scores (e.g. p-values) become larger
    assert _invert_scores(np.array([0.01])) > _invert_scores(np.array([0.1]))


def test_filter_by(liana_res: DataFrame) -> None:
    liana_res["interaction"] = liana_res["ligand_complex"] + " -> " + liana_res["receptor_complex"]

    assert _filter_by(liana_res, filter_fn=None) is liana_res

    filtered = _filter_by(liana_res, filter_fn=lambda x: x["specificity_rank"] > 0.95)
    # an interaction is kept whenever *any* of its cell type pairs passes
    kept = np.unique(liana_res[liana_res["specificity_rank"] > 0.95]["interaction"])
    assert set(filtered["interaction"]) == set(kept)


def test_get_top_n(liana_res: DataFrame) -> None:
    liana_res["interaction"] = liana_res["ligand_complex"] + " -> " + liana_res["receptor_complex"]

    assert _get_top_n(liana_res, top_n=None, orderby=None, orderby_ascending=None, orderby_absolute=False) is liana_res

    top = _get_top_n(
        liana_res.copy(), top_n=5, orderby="specificity_rank", orderby_ascending=False, orderby_absolute=False
    )
    assert top["interaction"].nunique() == 5
    # the categories carry the order the interactions were ranked in
    assert isinstance(top["interaction"].dtype, pd.CategoricalDtype)

    with pytest.raises(ValueError, match="specify the column to order"):
        _get_top_n(liana_res, top_n=5, orderby=None, orderby_ascending=False, orderby_absolute=False)

    with pytest.raises(ValueError, match="specify if `orderby` is ascending"):
        _get_top_n(liana_res, top_n=5, orderby="specificity_rank", orderby_ascending=None, orderby_absolute=False)
