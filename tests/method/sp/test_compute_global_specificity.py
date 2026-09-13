import pytest
from anndata import AnnData
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from liana.method import compute_global_specificity


def test_compute_global_specificity(pbmc68k: AnnData) -> None:
    compute_global_specificity(
        adata=pbmc68k, groupby="bulk_labels", lr_sep=None, n_perms=1, uns_key="global_interactions"
    )

    assert "global_interactions" in pbmc68k.uns
    res = pbmc68k.uns["global_interactions"]
    assert list(res.columns) == ["index", "feature", "lr_mean", "pval"]
    assert res["pval"].between(0.0, 1.0).all()


def test_raises_if_invalid_groupby(pbmc68k: AnnData) -> None:
    with pytest.raises(KeyError):
        compute_global_specificity(adata=pbmc68k, groupby="notagroup", n_perms=1, lr_sep=None)


def _run(adata: AnnData) -> DataFrame:
    compute_global_specificity(
        adata=adata, groupby="bulk_labels", lr_sep=None, n_perms=10, seed=1337, n_jobs=1, uns_key="gi"
    )
    res: DataFrame = adata.uns["gi"].copy()
    # `index` holds the group labels; cast away the categorical so the sort is on the labels
    # themselves rather than on whatever category order the run happened to use.
    res["index"] = res["index"].astype(str)
    return res.set_index(["index", "feature"]).sort_index()


def test_pvals_invariant_to_category_order(pbmc68k: AnnData) -> None:
    """`bulk_labels`' own category order is not lexicographic; both runs must agree."""
    reordered = pbmc68k.copy()
    reordered.obs["bulk_labels"] = reordered.obs["bulk_labels"].cat.reorder_categories(
        sorted(pbmc68k.obs["bulk_labels"].cat.categories)
    )

    assert_frame_equal(_run(pbmc68k), _run(reordered))
    # the stored labels keep the input's categorical dtype and its (non-alphabetical) category order
    assert list(pbmc68k.uns["gi"]["index"].cat.categories) == list(pbmc68k.obs["bulk_labels"].cat.categories)


def test_unused_category_is_dropped(pbmc68k: AnnData) -> None:
    """An unused category (e.g. left by a `min_cells` drop) must not shift or break the alignment."""
    with_unused = pbmc68k.copy()
    with_unused.obs["bulk_labels"] = with_unused.obs["bulk_labels"].cat.add_categories(["@unused"])

    assert_frame_equal(_run(pbmc68k), _run(with_unused))
