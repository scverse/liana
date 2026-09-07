from __future__ import annotations

import pandas as pd
import plotnine as p9

from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._types import Aggregator, RowFilter, SortKey
from liana.method.sp._misty._Misty import MistyData


@d.dedent
def misty_target_metrics(
    misty: MistyData | None = None,
    stat: str | None = None,
    target_metrics: pd.DataFrame | None = None,
    top_n: int | None = None,
    ascending: bool = False,
    key: SortKey | None = None,
    filter_fn: RowFilter | None = None,
    figure_size: tuple[float, float] = (5, 5),
    aggregate_fn: Aggregator | None = None,
    return_fig: bool = V.return_fig,
) -> p9.ggplot | None:
    """
    Plot target metrics.

    Parameters
    ----------
    %(misty)s
    stat
        Statistic to plot
    target_metrics
        A target_metrics DataFrame
    %(top_n)s
    ascending
        Whether to sort in ascending order
    key
        Function to use to sort the dataframe
    %(filter_fn)s
    %(aggregate_fn)s
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    Returns a plotnine plot.

    Raises
    ------
    ValueError
        If the `misty` input or `stat` provided are not valid.

    Examples
    --------
    Fit a MiSTy model, then plot one bar per target. `stat` chooses which of the
    `misty.uns['target_metrics']` columns is shown, and orders the targets by it:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> adata = adata[:, adata.var_names[:5]].copy()
    >>> misty = li.mt.genericMistyData(intra=adata, bandwidth=200, set_diag=True)
    >>> misty(model=li.mt.sp.LinearModel)
    >>> p = li.pl.misty_target_metrics(misty, stat="gain_R2")

    Pass `aggregate_fn` (e.g. `numpy.mean`) to summarise a masked model's
    per-group rows into boxplots.
    """
    if target_metrics is not None:
        target_metrics = target_metrics.copy()
    elif misty is not None:
        target_metrics = misty.uns[K.target_metrics].copy()
    else:
        raise ValueError("Provide either a misty object or a target_metrics DataFrame.")
    if stat is None:
        raise ValueError("Provide a statistic to plot")

    if filter_fn is not None:
        target_metrics = target_metrics[target_metrics.apply(filter_fn, axis=1).astype(bool)]
    if aggregate_fn is not None:
        targets = target_metrics.groupby(["target"]).agg({stat: aggregate_fn})
        targets = targets.sort_values(by=stat, ascending=ascending).index
    else:
        targets = target_metrics.sort_values(by=stat, ascending=ascending, key=key)["target"].unique()
    if top_n is not None:
        target_metrics = target_metrics[target_metrics["target"].isin(targets[:top_n])]

    # keep order of targets
    target_metrics["target"] = pd.Categorical(target_metrics["target"], categories=targets, ordered=True)

    p = (
        p9.ggplot(target_metrics, p9.aes(x="target", y=stat))
        + (p9.geom_boxplot() if aggregate_fn is not None else p9.geom_point(size=3))
        + p9.theme_bw()
        + p9.theme(axis_text_x=p9.element_text(rotation=90), figure_size=figure_size)
        + p9.labs(x="Target")
    )

    if return_fig:
        return p
    p.draw()
    return None


@d.dedent
def misty_contributions(
    misty: MistyData | None = None,
    target_metrics: pd.DataFrame | None = None,
    view_names: list[str] | None = None,
    filter_fn: RowFilter | None = None,
    aggregate_fn: Aggregator | None = None,
    figure_size: tuple[float, float] = (5, 5),
    return_fig: bool = V.return_fig,
) -> p9.ggplot | None:
    """
    Plot view contributions per target.

    Parameters
    ----------
    %(misty)s
    target_metrics
        A target_metrics DataFrame
    view_names
        A list of view names to plot
    %(filter_fn)s
    %(aggregate_fn)s
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A plotnine plot.

    Raises
    ------
    ValueError
        If the `misty` input is not valid or no `view_names` provided.

    Examples
    --------
    Stacks each view's contribution to every target into one bar, read from
    `misty.uns['target_metrics']`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> adata = adata[:, adata.var_names[:5]].copy()
    >>> misty = li.mt.genericMistyData(intra=adata, bandwidth=200, set_diag=True)
    >>> misty(model=li.mt.sp.LinearModel)
    >>> p = li.pl.misty_contributions(misty)
    """
    if target_metrics is not None:
        target_metrics = target_metrics.copy()
    elif misty is not None:
        target_metrics = misty.uns[K.target_metrics].copy()
    else:
        raise ValueError("Provide either a misty object or a target_metrics DataFrame.")

    if view_names is None:
        if misty is None:
            raise ValueError("Provide a list of view names to plot.")
        view_names = misty.view_names.copy()
        if "intra" not in target_metrics.columns:
            view_names.remove("intra")

    if filter_fn is not None:
        target_metrics = target_metrics[target_metrics.apply(filter_fn, axis=1).astype(bool)]
        # NOTE: `target` is only categorical when it comes from an aggregated table
        if isinstance(target_metrics["target"].dtype, pd.CategoricalDtype):
            target_metrics["target"] = target_metrics["target"].cat.remove_unused_categories()

    target_metrics = target_metrics[["target", *view_names]]
    target_metrics = target_metrics.melt(id_vars="target", var_name="view", value_name="contribution")

    if aggregate_fn is not None:
        target_metrics = target_metrics.groupby(["target", "view"]).agg({"contribution": aggregate_fn}).reset_index()

    p = (
        p9.ggplot(target_metrics, p9.aes(x="target", y="contribution", fill="view"))
        + p9.geom_bar(stat="identity")
        + p9.theme_bw(base_size=14)
        + p9.theme(axis_text_x=p9.element_text(rotation=90), figure_size=figure_size)
        + p9.scale_fill_brewer(palette=2, type="qual")
        + p9.labs(x="Target", y="Contribution", fill="View")
    )

    if return_fig:
        return p
    p.draw()
    return None


@d.dedent
def misty_interactions(
    misty: MistyData | None = None,
    interactions: pd.DataFrame | None = None,
    view: str | None = None,
    top_n: int | None = None,
    ascending: bool = False,
    key: SortKey | None = None,
    filter_fn: RowFilter | None = None,
    aggregate_fn: Aggregator | None = None,
    figure_size: tuple[float, float] = (5, 5),
    return_fig: bool = V.return_fig,
) -> p9.ggplot | None:
    """
    Plot interaction importances.

    Parameters
    ----------
    %(misty)s
    interactions
        Interactions from which to plot the importance
    view
        A Misty view to plot
    %(top_n)s
    ascending
        Whether to sort interactions in ascending order
    key
        Function to use to sort the interactions
    %(filter_fn)s
    %(aggregate_fn)s
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A plotnine plot.

    Raises
    ------
    ValueError
        If neither `misty` or `interactions` input or `view` are not provided.

    Examples
    --------
    One tile per predictor-target pair of the chosen `view`, coloured by the
    importance the model gave it, read from `misty.uns['interactions']`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> adata = adata[:, adata.var_names[:5]].copy()
    >>> misty = li.mt.genericMistyData(intra=adata, bandwidth=200, set_diag=True)
    >>> misty(model=li.mt.sp.LinearModel)
    >>> p = li.pl.misty_interactions(misty, view="intra")

    `misty.view_names` lists the views available -- here `'intra'`, `'juxta'` and
    `'para'`.
    """
    if interactions is not None:
        interactions = interactions.copy()
    elif misty is not None:
        interactions = misty.uns[K.interactions].copy()
    else:
        raise ValueError("Provide either a misty object or interactions.")
    if view is None:
        raise ValueError("Provide a ``view`` to plot!")

    interactions = interactions[interactions["view"] == view]
    grouped = interactions.groupby("predictor")["importances"].apply(lambda x: x.isna().all())
    interactions = interactions[~interactions["predictor"].isin(grouped[grouped].index)]

    if filter_fn is not None:
        interactions = interactions[interactions.apply(filter_fn, axis=1).astype(bool)]
    if aggregate_fn is not None:
        interactions = interactions.groupby(["target", "predictor"]).agg({"importances": aggregate_fn}).reset_index()
    if top_n is not None:
        interactions = interactions.sort_values(by="importances", key=key, ascending=ascending)
        top_interactions = interactions.drop_duplicates(["target", "predictor"]).head(top_n)
        interactions = interactions[
            interactions["target"].isin(top_interactions["target"])
            & interactions["predictor"].isin(top_interactions["predictor"])
        ]

    p = (
        p9.ggplot(interactions, p9.aes(x="predictor", y="target", fill="importances"))
        + p9.geom_tile()
        + p9.theme_minimal(base_size=12)
        + p9.theme(axis_text_x=p9.element_text(rotation=90), figure_size=figure_size)
        + p9.labs(x="Predictor", y="Target", fill="Importance")
    )

    if return_fig:
        return p
    p.draw()
    return None
