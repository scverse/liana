import numpy as np
import pytest
from anndata import AnnData
from matplotlib.axes import Axes
from pandas import DataFrame, Series
from tests._helpers import invalid

from liana.plotting import circle
from liana.plotting._circle_plot import (
    _get_adata_colors,
    _pivot_liana_res,
    _scale_list,
)
from liana.plotting._common import _filter_by, _get_top_n


@pytest.fixture
def adata(pbmc68k: AnnData, liana_res: DataFrame) -> AnnData:
    """`pbmc68k` labelled with the cell types of the toy liana results."""
    rng = np.random.default_rng(0)
    pbmc68k.obs["random"] = rng.choice(np.unique(liana_res["source"]), size=pbmc68k.shape[0])
    pbmc68k.uns["liana_res"] = liana_res

    return pbmc68k


def test_circle_plot(adata: AnnData, liana_res: DataFrame) -> None:
    # circle_plot returns bare Axes, so assert on the adjacency it is built from:
    # 'mean' averages the score per source-target pair over the rows that pass `filter_fn` ...
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

    # ... and `mask_mode='and'` on a valid pair that has no interactions
    source, target = liana_res["source"].iloc[0], liana_res["target"].iloc[0]
    adata.uns["liana_res"] = liana_res[~(liana_res["source"].eq(source) & liana_res["target"].eq(target))]
    with pytest.raises(ValueError, match="No interactions remain to plot"):
        circle(adata, groupby="random", source_labels=[source], target_labels=[target], mask_mode="and")


def _counts(ax: Axes) -> tuple[int, int]:
    """The number of nodes and of edges drawn: the node scatter is the last collection."""
    return np.asarray(ax.collections[-1].get_offsets()).shape[0], len(ax.patches)


def test_circle_labels_filter_rows(adata: AnnData, liana_res: DataFrame) -> None:
    """The labels filter the interactions before the pivot, which is then squared (#184, #187)."""
    source = liana_res["source"].iloc[0]
    targets_of = set(liana_res[liana_res["source"].eq(source)]["target"])

    # a source subset used to truncate to a single node, since its pivot is not square
    nodes, edges = _counts(circle(adata, groupby="random", liana_res=liana_res, source_labels=[source]))
    assert (nodes, edges) == (len(targets_of | {source}), len(targets_of))

    # ... and a target subset used to raise `Columns must match Indices`
    target = liana_res["target"].iloc[0]
    sources_of = set(liana_res[liana_res["target"].eq(target)]["source"])
    nodes, edges = _counts(circle(adata, groupby="random", liana_res=liana_res, target_labels=[target]))
    assert (nodes, edges) == (len(sources_of | {target}), len(sources_of))


def test_circle_filter_fn_is_row_wise_in_mean_mode(adata: AnnData, liana_res: DataFrame) -> None:
    """A 'mean' edge averages the pairs that pass `filter_fn`, not every pair of a passing interaction."""
    res = liana_res.assign(interaction=liana_res["ligand_complex"] + " -> " + liana_res["receptor_complex"])
    # an interaction with several sources, passing the filter through one of them only
    sources = res.groupby("interaction")["source"].unique()
    interaction, labels = next((name, group) for name, group in sources.items() if len(group) > 1)
    source, other = labels[:2]

    def filter_fn(row: Series) -> bool:
        return bool(row["interaction"] == interaction and row["source"] == other)

    # interaction-wise selection (what `dotplot` does) would keep `source`'s pairs too ...
    assert _filter_by(res, filter_fn)["source"].eq(source).any()
    # ... while the circle plot keeps only the rows that pass, so `source` carries no edge
    passing = res[res.apply(filter_fn, axis=1)]
    expected = _pivot_liana_res(passing, score_key="specificity_rank", mode="mean")
    ax = circle(
        adata,
        groupby="random",
        liana_res=liana_res,
        pivot_mode="mean",
        score_key="specificity_rank",
        filter_fn=filter_fn,
    )
    assert _counts(ax)[1] == int((expected.to_numpy() != 0).sum())
    assert source not in expected.index

    # and asking for `source` on top of that filter leaves nothing to draw
    with pytest.raises(ValueError, match="No interactions remain to plot"):
        circle(
            adata,
            groupby="random",
            liana_res=liana_res,
            pivot_mode="mean",
            score_key="specificity_rank",
            filter_fn=filter_fn,
            source_labels=[source],
        )


def test_circle_top_n_ranks_within_labels(adata: AnnData, liana_res: DataFrame) -> None:
    """`source_labels` narrow the frame before `top_n`, so the top interactions are the source's own."""
    res = liana_res.assign(interaction=liana_res["ligand_complex"] + " -> " + liana_res["receptor_complex"])
    source = res["source"].iloc[0]
    own = res[res["source"].eq(source)]
    expected = _pivot_liana_res(
        _get_top_n(own, top_n=1, orderby="specificity_rank", orderby_ascending=True, orderby_absolute=False),
        mode="counts",
    )
    ax = circle(
        adata,
        groupby="random",
        liana_res=liana_res,
        source_labels=[source],
        top_n=1,
        orderby="specificity_rank",
        orderby_ascending=True,
    )
    assert _counts(ax)[1] == int((expected.to_numpy() != 0).sum())
    # ... whereas the global top interaction need not involve `source` at all
    global_top = _get_top_n(res, top_n=1, orderby="specificity_rank", orderby_ascending=True, orderby_absolute=False)
    assert not global_top["source"].eq(source).any()


def test_circle_mask_mode(adata: AnnData, liana_res: DataFrame) -> None:
    """`'and'` intersects the two label sets, `'or'` unions them."""
    source, target = "A", "B"
    both = circle(
        adata,
        groupby="random",
        liana_res=liana_res,
        source_labels=[source],
        target_labels=[target],
        mask_mode="and",
    )
    assert _counts(both) == (2, 1)

    either = circle(
        adata,
        groupby="random",
        liana_res=liana_res,
        source_labels=[source],
        target_labels=[target],
        mask_mode="or",
    )
    assert _counts(either)[1] > _counts(both)[1]


def test_circle_uses_passed_liana_res(adata: AnnData, liana_res: DataFrame) -> None:
    """`adata` is required here, so `.uns` must not shadow an explicit frame (#187)."""
    subset = liana_res[liana_res["source"].isin(["A", "B"])]
    assert _counts(circle(adata, groupby="random", liana_res=subset)) != _counts(
        circle(adata, groupby="random", liana_res=liana_res)
    )


def test_circle_keeps_zero_weight_edges(adata: AnnData, liana_res: DataFrame) -> None:
    """A pair whose rows average to exactly 0 is still an edge; only pairs without rows are absent."""
    res = liana_res.copy()
    source, target = "A", "B"
    pair = res["source"].eq(source) & res["target"].eq(target)
    assert pair.any()
    # `-log10(1) == 0`: an inverted rank of 1 used to be indistinguishable from "no interactions"
    res.loc[pair, "magnitude_rank"] = 1.0
    res.loc[~pair, "magnitude_rank"] = 0.5

    with_zero = circle(
        adata, groupby="random", liana_res=res, pivot_mode="mean", score_key="magnitude_rank", inverse_score=True
    )
    without = circle(
        adata, groupby="random", liana_res=res[~pair], pivot_mode="mean", score_key="magnitude_rank", inverse_score=True
    )
    assert _counts(with_zero)[1] == _counts(without)[1] + 1
