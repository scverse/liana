from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotnine as p9
from anndata import AnnData

from liana._core._common import _get_liana_res
from liana._core._constants import DefaultValues as V
from liana._core._docs import d
from liana.method.sp._lric_helpers import CurveTransform, _log2_floor, _mean_curve, get_lric_divergence

_ID_COLS = ("source", "target", "ligand_complex", "receptor_complex", "interaction")


def _preview(values: Iterable[object], n: int = 10) -> str:
    unique = list(dict.fromkeys(map(str, values)))
    return ", ".join(unique[:n]) + (f", ... ({len(unique)} total)" if len(unique) > n else "")


@d.dedent
def lric_lineplot(
    adata: AnnData | None = None,
    uns_key: str = "lric",
    liana_res: pd.DataFrame | None = None,
    interaction: str | None = None,
    source: str | None = None,
    target: str | None = None,
    max_dist: float | None = None,
    figure_size: tuple[float, float] = (6, 4),
    return_fig: bool = V.return_fig,
) -> p9.ggplot | None:
    """
    Plot the g(r) profile of a single interaction from ``lric`` / ``cross_pcf``.

    ``g(r)`` is the observed count at radius ``r`` relative to the closed-form
    random-labelling null; ``g(r) > 1`` is co-enrichment, ``< 1`` depletion, and
    the dashed line at ``1`` marks the null. For pairwise LRIC the full coupling
    is decomposed into its architecture-only (``g_pcf``) and expression-only
    (``g_expr``) components, drawn as separate lines.

    Parameters
    ----------
    %(adata)s
        Its ``.uns[uns_key]`` holds the result. Mutually exclusive with
        ``liana_res``.
    %(uns_key)s
    %(liana_res)s
        A ``lric`` / ``cross_pcf`` result, used when ``adata`` is ``None``.
    interaction
        The ``interaction`` to plot -- ``"source^target"`` for ``cross_pcf``,
        ``"ligand^receptor"`` for ``lric``.
    source
        Sender cell type; only for results that carry a ``source`` column.
    target
        Receiver cell type; only for results that carry a ``target`` column.
    max_dist
        Plot only radii ``r < max_dist``; ``None`` uses all radii.
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A ``plotnine.ggplot`` if ``return_fig`` else ``None`` (draws the plot).
    """
    res = _get_liana_res(adata, liana_res, uns_key)
    ids = [c for c in _ID_COLS if c in res.columns]

    for col, value in (("interaction", interaction), ("source", source), ("target", target)):
        if value is None:
            continue
        if col not in ids:
            raise ValueError(f"`{col}` is not a column of this result; it has {ids}.")
        subset = res[res[col] == value]
        if subset.empty:
            raise ValueError(f"`{col}={value!r}` not found. Available: {_preview(res[col])}")
        res = subset

    combos = res[ids].drop_duplicates()
    if len(combos) != 1:
        raise ValueError(
            f"The selection resolves to {len(combos)} interactions, expected exactly one. "
            f"Narrow it down with {[c for c in ids if c in ('interaction', 'source', 'target')]}; "
            f"candidates: {_preview(combos.astype(str).agg(' | '.join, axis=1))}"
        )

    row = combos.iloc[0]
    title = " -> ".join(str(row[c]) for c in ("source", "target") if c in ids)
    if "ligand_complex" in ids:
        title = f"{title}: {row['interaction']}" if title else str(row["interaction"])

    curves = {"g": "g (full)", "g_pcf": "g_pcf", "g_expr": "g_expr"} if "g_pcf" in res.columns else {"g": "g(r)"}
    res = res.sort_values("radius")
    if max_dist is not None:
        in_window = res[res["radius"] < max_dist]
        if in_window.empty:
            raise ValueError(f"No radii below `max_dist={max_dist}`; the grid is {_preview(res['radius'].unique())}.")
        res = in_window
    df = pd.concat(
        [
            pd.DataFrame({"radius": res["radius"].to_numpy(float), "g": res[col].to_numpy(float), "curve": label})
            for col, label in curves.items()
        ],
        ignore_index=True,
    )
    df["curve"] = pd.Categorical(df["curve"], categories=list(curves.values()))

    p = (
        p9.ggplot(df, p9.aes("radius", "g", color="curve"))
        + p9.geom_hline(yintercept=1, linetype="dashed", color="grey")
        + p9.geom_line()
        + p9.geom_point(size=1.2)
        + p9.labs(x="Radius (r)", y="g(r)", title=title, color="")
        + p9.theme_bw()
        + p9.theme(figure_size=figure_size)
    )
    if len(curves) == 1:  # single curve -> the legend adds nothing
        p = p + p9.theme(legend_position="none")

    if return_fig:
        return p

    p.draw()
    return None


@d.dedent
def lric_divergence(
    adata: AnnData | None = None,
    uns_key: str = "lric",
    liana_res: pd.DataFrame | None = None,
    feature_a: dict[str, object] | None = None,
    feature_b: dict[str, object] | None = None,
    max_dist: float | None = None,
    transform_fn: CurveTransform = _log2_floor,
    min_bins: int = 3,
    figure_size: tuple[float, float] = (6, 4),
    return_fig: bool = V.return_fig,
) -> p9.ggplot | None:
    """
    Plot two ``transform_fn(g(r))`` curves and the area between them.

    The visual companion of :func:`liana.mt.get_lric_divergence`: both curves
    are drawn over radius, the grey ribbon spans their separation, the dashed
    line at ``0`` marks the null, and the dotted vertical line marks ``r_star``
    -- the radius where the curves diverge most. The title carries the
    span-normalised divergence and ``r_star``.

    Parameters
    ----------
    %(adata)s
        Its ``.uns[uns_key]`` holds the result. Mutually exclusive with
        ``liana_res``.
    %(uns_key)s
    %(liana_res)s
        A ``lric`` / ``cross_pcf`` result, used when ``adata`` is ``None``.
    feature_a
        Selection of the first curve as ``{column: value}`` over any columns of
        the result, e.g. ``dict(interaction="Dcn^Egfr")``. It must resolve to a
        single interaction; see :func:`liana.mt.get_lric_divergence`.
    feature_b
        Selection of the second curve; same rules as ``feature_a``.
    max_dist
        Compare and draw only radii ``r < max_dist``; ``None`` uses all radii.
    transform_fn
        Applied to ``g`` before comparing; defaults to log2 with ``g`` floored
        at ``0.05``. Pass :obj:`numpy.log2` to drop non-finite bins instead.
    min_bins
        Minimum shared finite radius bins required; fewer raises a ``ValueError``.
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    A ``plotnine.ggplot`` if ``return_fig`` else ``None`` (draws the plot).

    Examples
    --------
    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> li.mt.cross_pcf(adata, groupby="bulk_labels", key_added="cross_pcf")
    >>> p = li.pl.lric_divergence(
    ...     adata,
    ...     "cross_pcf",
    ...     feature_a=dict(source="CD14+ Monocyte", target="CD34+"),
    ...     feature_b=dict(source="CD14+ Monocyte", target="CD19+ B"),
    ... )
    """
    res = _get_liana_res(adata, liana_res, uns_key)
    if not feature_a or not feature_b:
        raise ValueError("`feature_a` and `feature_b` selections must be provided!")
    div = get_lric_divergence(
        liana_res=res,
        feature_a=feature_a,
        feature_b=feature_b,
        max_dist=max_dist,
        transform_fn=transform_fn,
        min_bins=min_bins,
    )

    ids = [c for c in _ID_COLS if c in res.columns]
    curves = pd.concat(
        [_mean_curve(res, feature_a, ids, transform_fn), _mean_curve(res, feature_b, ids, transform_fn)],
        axis=1,
        keys=[div["label_a"], div["label_b"]],
    )
    curves = curves[np.isfinite(curves).all(axis=1)]
    if max_dist is not None:
        curves = curves[curves.index < max_dist]

    ribbon = pd.DataFrame(
        {
            "radius": curves.index,
            "ymin": curves.min(axis=1),
            "ymax": curves.max(axis=1),
        }
    )
    df = curves.rename_axis("radius").reset_index().melt(id_vars="radius", var_name="curve", value_name="g")

    ylab = "log2 g(r)" if transform_fn in (_log2_floor, np.log2) else "transform(g(r))"
    p = (
        p9.ggplot(df, p9.aes("radius", "g", color="curve"))
        + p9.geom_ribbon(
            p9.aes(x="radius", ymin="ymin", ymax="ymax"), data=ribbon, fill="grey", alpha=0.35, inherit_aes=False
        )
        + p9.geom_hline(yintercept=0, linetype="dashed", color="grey")
        + p9.geom_vline(xintercept=div["r_star"], linetype="dotted", color="black")
        + p9.geom_line()
        + p9.geom_point(size=1.2)
        + p9.labs(x="Radius (r)", y=ylab, title=f"divergence={div['divergence']:.3g}  r*={div['r_star']:g}", color="")
        + p9.theme_bw()
        + p9.theme(figure_size=figure_size)
    )

    if return_fig:
        return p

    p.draw()
    return None
