import numpy as np
import pytest
from anndata import AnnData
from pandas import DataFrame
from tests._helpers import invalid

from liana.plotting import circle
from liana.plotting._circle_plot import (
    _get_adata_colors,
    _pivot_liana_res,
    _scale_list,
    get_mask_df,
)


@pytest.fixture
def adata(pbmc68k: AnnData, liana_res: DataFrame) -> AnnData:
    """`pbmc68k` labelled with the cell types of the toy liana results."""
    rng = np.random.default_rng(0)
    pbmc68k.obs["random"] = rng.choice(np.unique(liana_res["source"]), size=pbmc68k.shape[0])
    pbmc68k.uns["liana_res"] = liana_res

    return pbmc68k


def test_circle_plot(adata: AnnData, liana_res: DataFrame) -> None:
    # circle_plot returns bare Axes, so assert on the adjacency it is built from:
    # 'mean' averages the score per source-target pair ...
    circle(adata, groupby="random", liana_res=liana_res, pivot_mode="mean", score_key="specificity_rank")
    means = _pivot_liana_res(liana_res, score_key="specificity_rank", mode="mean")
    expected = liana_res.groupby(["source", "target"])["specificity_rank"].mean()
    assert means.loc["A", "B"] == pytest.approx(expected.loc["A", "B"])
    assert set(means.index) == set(liana_res["source"])
    assert set(means.columns) == set(liana_res["target"])

    # ... while 'counts' counts the interactions surviving the filter
    circle(
        adata,
        groupby="random",
        liana_res=liana_res,
        pivot_mode="counts",
        filter_fn=lambda x: x["specificity_rank"] < 0.95,
    )
    kept = liana_res[liana_res["specificity_rank"] < 0.95]
    counts = _pivot_liana_res(kept, mode="counts")
    assert counts.loc["A", "B"] == (kept["source"].eq("A") & kept["target"].eq("B")).sum()
    assert counts.to_numpy().sum() == kept.shape[0]


def test_circle_plot_raises(adata: AnnData, liana_res: DataFrame) -> None:
    with pytest.raises(ValueError, match="`groupby` must be provided"):
        circle(adata, groupby=None, liana_res=liana_res)

    with pytest.raises(ValueError, match="`pivot_mode` must be 'counts' or 'mean'"):
        circle(adata, groupby="random", liana_res=liana_res, pivot_mode=invalid("neither"))

    with pytest.raises(ValueError, match="`score_key` must be provided"):
        circle(adata, groupby="random", liana_res=liana_res, pivot_mode="mean", score_key=None)

    with pytest.raises(KeyError, match="`not_a_column` not found in `adata.obs.columns`"):
        circle(adata, groupby="not_a_column", liana_res=liana_res)


def test_get_mask_df(liana_res: DataFrame) -> None:
    pivot_table = _pivot_liana_res(liana_res, score_key="specificity_rank", mode="mean")
    source, target = sorted(pivot_table.index)[:2]

    # nothing to mask by
    assert get_mask_df(pivot_table) is pivot_table

    # 'or' keeps a full row and a full column, 'and' only their intersection
    either = get_mask_df(pivot_table.copy(), source_cell_type=source, target_cell_type=target, mode="or")
    both = get_mask_df(pivot_table.copy(), source_cell_type=source, target_cell_type=target, mode="and")
    assert (either != 0).sum().sum() > (both != 0).sum().sum()
    assert (both.drop(index=source) == 0).all().all()
    assert (both.drop(columns=target) == 0).all().all()


def test_get_adata_colors_defaults(adata: AnnData) -> None:
    # a colour per category is assigned by default
    defaults = _get_adata_colors(adata, "random")
    assert set(defaults) == set(adata.obs["random"].unique())


def test_scale_list_degenerate_range(adata: AnnData, liana_res: DataFrame) -> None:
    # a single value, or all-equal values, has no range to scale into - `0/0` would make the
    # one edge the user asked for NaN-wide, i.e. invisible
    np.testing.assert_array_equal(_scale_list([4.0], min_val=1, max_val=5), [5.0])
    np.testing.assert_array_equal(_scale_list([2.0, 2.0, 2.0], min_val=1, max_val=5), [5.0, 5.0, 5.0])

    # ... as reached by masking down to a single source-target pair
    source, target = liana_res["source"].iloc[0], liana_res["target"].iloc[0]
    ax = circle(
        adata,
        groupby="random",
        liana_res=liana_res,
        source_labels=[source],
        target_labels=[target],
        mask_mode="and",
    )
    assert len(ax.patches) == 1
    assert not np.isnan([patch.get_linewidth() for patch in ax.patches]).any()


def test_circle_does_not_mutate_adata(adata: AnnData, liana_res: DataFrame) -> None:
    # the default-palette branch: neither the categorical conversion nor `{label}_colors`
    # may be written back to the caller's object
    dtype, uns_keys = adata.obs["random"].dtype, set(adata.uns)
    circle(adata, groupby="random", liana_res=liana_res)
    assert adata.obs["random"].dtype == dtype
    assert set(adata.uns) == uns_keys

    # ... and the branch where a palette is already there, which still needs the categories
    adata.uns["random_colors"] = list(_get_adata_colors(adata, "random").values())
    dtype, uns_keys = adata.obs["random"].dtype, set(adata.uns)
    circle(adata, groupby="random", liana_res=liana_res)
    assert adata.obs["random"].dtype == dtype
    assert set(adata.uns) == uns_keys


def test_circle_no_edges_raises(adata: AnnData, liana_res: DataFrame) -> None:
    """Zero edges used to reach `_scale_list`'s `np.min([])`."""
    # an over-restrictive filter, which leaves no interactions at all ...
    with pytest.raises(ValueError, match="No interactions remain to plot"):
        circle(adata, groupby="random", liana_res=liana_res, filter_fn=lambda x: False)

    # ... and `mask_mode='and'` on a pair that has none, which keeps the nodes but no edge
    source, target = liana_res["source"].iloc[0], liana_res["target"].iloc[0]
    adata.uns["liana_res"] = liana_res[~(liana_res["source"].eq(source) & liana_res["target"].eq(target))]
    with pytest.raises(ValueError, match="No interactions remain to plot"):
        circle(adata, groupby="random", source_labels=[source], target_labels=[target], mask_mode="and")
