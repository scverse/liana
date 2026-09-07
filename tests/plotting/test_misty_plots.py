from contextlib import suppress

import pandas as pd
import pytest
from anndata import AnnData
from pandas import DataFrame
from tests._helpers import get_obs
from tests._helpers import plot_data as _frame

import liana.plotting as pl
from liana.datasets._sample_misty import _sample_interactions, _sample_target_metrics
from liana.method.sp import MistyData


@pytest.fixture
def misty(toy_spatial: AnnData) -> MistyData:
    """A two-view `MistyData` carrying toy misty results.

    A real `MistyData` rather than an AnnData wearing a `view_names` attribute, so that it matches what the plotting functions declare they take.
    """
    views = {"intra": toy_spatial.copy(), "extra": toy_spatial.copy()}
    misty = MistyData(views, obs=get_obs(toy_spatial))
    misty.uns["target_metrics"] = _sample_target_metrics(seed=0)
    misty.uns["interactions"] = _sample_interactions(seed=0)

    return misty


@pytest.fixture
def target_metrics() -> DataFrame:
    """Target metrics for two groups, e.g. two samples to aggregate over."""
    target_metrics = pd.concat([_sample_target_metrics(seed=0), _sample_target_metrics(seed=1)])
    target_metrics["group"] = ["a"] * 3 + ["b"] * 3

    return target_metrics


@pytest.fixture
def interactions() -> DataFrame:
    """Interactions for two groups, e.g. two samples to aggregate over."""
    interactions = pd.concat([_sample_interactions(seed=0), _sample_interactions(seed=1)])
    interactions["group"] = ["a"] * 9 + ["b"] * 9

    return interactions


def test_target_contributions_plot(misty: MistyData) -> None:
    plot_data = _frame(pl.misty_contributions(misty=misty))

    # melted to one row per target x view, with no target or view dropped
    targets = misty.uns["target_metrics"]["target"]
    assert set(plot_data["target"]) == set(targets)
    assert set(plot_data["view"]) == {"intra", "extra"}
    assert plot_data.shape[0] == len(targets) * 2

    # `return_fig=False` draws the plot instead of handing it back
    assert pl.misty_contributions(misty=misty, return_fig=False) is None


def test_target_metrics_plot(misty: MistyData) -> None:
    target_metrics = misty.uns["target_metrics"]

    plot_data = _frame(pl.misty_target_metrics(misty=misty, stat="gain_R2"))
    assert set(plot_data["target"]) == set(target_metrics["target"])

    # top_n keeps the n best targets by the statistic asked for
    top = _frame(pl.misty_target_metrics(misty=misty, stat="gain_R2", top_n=1))
    best = target_metrics.loc[target_metrics["gain_R2"].idxmax(), "target"]
    assert set(top["target"]) == {best}

    filtered = _frame(pl.misty_target_metrics(misty=misty, stat="gain_R2", filter_fn=lambda x: x["multi_R2"] > 0.5))
    expected = target_metrics[target_metrics["multi_R2"] > 0.5]["target"]
    assert 0 < len(expected) < len(target_metrics)  # else the check below is vacuous
    assert set(filtered["target"]) == set(expected)

    assert pl.misty_target_metrics(misty=misty, stat="gain_R2", return_fig=False) is None


def test_interactions_plot(misty: MistyData, interactions: DataFrame) -> None:
    pl.misty_interactions(misty=misty, top_n=3, view="extra", key=abs, ascending=False)
    plot_data = _frame(
        pl.misty_interactions(interactions=interactions, view="extra", filter_fn=lambda x: x["group"] == "b")
    )
    assert plot_data.shape[0] == 3

    assert pl.misty_interactions(misty=misty, view="extra", return_fig=False) is None


def test_target_metrics_aggregate(target_metrics: DataFrame) -> None:
    plot_data = _frame(pl.misty_target_metrics(target_metrics=target_metrics, stat="gain_R2", aggregate_fn="mean"))

    # every group is kept - aggregation only decides the order the targets
    # are drawn in, best mean first
    assert plot_data.shape[0] == target_metrics.shape[0]
    expected = target_metrics.groupby("target")["gain_R2"].mean().sort_values(ascending=False).index.tolist()
    assert list(plot_data["target"].cat.categories) == expected


def test_contributions_aggregate(target_metrics: DataFrame) -> None:
    plot_data = _frame(
        pl.misty_contributions(target_metrics=target_metrics, view_names=["intra", "extra"], aggregate_fn="median")
    )

    n_targets = target_metrics["target"].nunique()
    assert plot_data.shape[0] == n_targets * 2
    expected = target_metrics.groupby("target")["intra"].median()
    actual = plot_data[plot_data["view"] == "intra"].set_index("target")["contribution"]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False, check_dtype=False)


def test_interactions_aggregate(interactions: DataFrame) -> None:
    plot_data = _frame(pl.misty_interactions(interactions=interactions, view="intra", aggregate_fn="sum"))

    # only the requested view is drawn, with importances summed over the groups
    intra = interactions[interactions["view"] == "intra"]
    assert set(plot_data["target"]) == set(intra["target"])
    expected = intra.groupby(["target", "predictor"])["importances"].sum()
    actual = plot_data.set_index(["target", "predictor"])["importances"]
    pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False, check_dtype=False)


def test_misty_plots_raise_without_data() -> None:
    with pytest.raises(ValueError, match="Provide either a misty object or a target_metrics"):
        pl.misty_target_metrics(stat="gain_R2")

    with pytest.raises(ValueError, match="Provide either a misty object or a target_metrics"):
        pl.misty_contributions(view_names=["intra", "extra"])

    with pytest.raises(ValueError, match="Provide either a misty object or interactions"):
        pl.misty_interactions(view="intra")


def test_misty_plots_raise_on_missing_args(
    misty: MistyData, target_metrics: DataFrame, interactions: DataFrame
) -> None:
    with pytest.raises(ValueError, match="Provide a statistic to plot"):
        pl.misty_target_metrics(misty=misty, stat=None)

    with pytest.raises(ValueError, match="Provide a list of view names to plot"):
        pl.misty_contributions(target_metrics=target_metrics)

    with pytest.raises(ValueError, match="Provide a ``view`` to plot"):
        pl.misty_interactions(interactions=interactions, view=None)


def test_contributions_filter(misty: MistyData) -> None:
    plot_data = _frame(pl.misty_contributions(misty=misty, filter_fn=lambda x: x["multi_R2"] > 0.5))

    target_metrics = misty.uns["target_metrics"]
    expected = target_metrics[target_metrics["multi_R2"] > 0.5]["target"]
    assert 0 < len(expected) < len(target_metrics)
    assert set(plot_data["target"]) == set(expected)


def test_contributions_drops_intra_when_absent(misty: MistyData) -> None:
    # `intra` is dropped from the views when it is not among the metrics
    del misty.uns["target_metrics"]["intra"]
    plot_data = _frame(pl.misty_contributions(misty=misty))
    assert set(plot_data["view"]) == {"extra"}


def test_contributions_filter_on_categorical_target(target_metrics: DataFrame) -> None:
    target_metrics["target"] = target_metrics["target"].astype("category")
    plot_data = _frame(
        pl.misty_contributions(
            target_metrics=target_metrics,
            view_names=["intra", "extra"],
            aggregate_fn="mean",
            filter_fn=lambda x: x["target"] != "a",
        )
    )
    assert "a" not in set(plot_data["target"].cat.categories)


@pytest.mark.parametrize(
    ("deprecated", "current"),
    [
        ("contributions", "misty_contributions"),
        ("interactions", "misty_interactions"),
        ("target_metrics", "misty_target_metrics"),
        ("circle_plot", "circle"),
        ("annulus_plot", "annulus"),
        ("lric_divergence_plot", "lric_divergence"),
    ],
)
def test_renamed_plots_still_resolve(deprecated: str, current: str) -> None:
    alias = getattr(pl, deprecated)

    assert f"Use `liana.pl.{current}` instead" in alias.__deprecated__
    assert not hasattr(getattr(pl, current), "__deprecated__")

    with pytest.warns(FutureWarning, match=f"{deprecated} is deprecated"), suppress(TypeError, ValueError):
        alias()
