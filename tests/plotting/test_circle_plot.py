import numpy as np
import pytest
from anndata import AnnData
from pandas import DataFrame
from tests._helpers import invalid

from liana.plotting import circle
from liana.plotting._circle_plot import _get_adata_colors, _pivot_liana_res, _set_adata_color, get_mask_df


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


def test_set_adata_color(adata: AnnData) -> None:
    # a colour per category is assigned by default
    _set_adata_color(adata, "random")
    defaults = _get_adata_colors(adata, "random")
    assert set(defaults) == set(adata.obs["random"].cat.categories)

    # ... and can be overridden, for named colours as well as hex ones
    _set_adata_color(adata, "random", color_dict={"A": "red"}, hex=False)
    assert _get_adata_colors(adata, "random")["A"] == "#ff0000"

    _set_adata_color(adata, "random", color_dict={"A": "#00ff00"})
    assert _get_adata_colors(adata, "random")["A"] == "#00ff00"
