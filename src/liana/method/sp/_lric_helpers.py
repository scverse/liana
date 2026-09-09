from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray

from liana._core._common import _get_liana_res, _logg
from liana._core._docs import d

_ID_COLS = ("source", "target", "ligand_complex", "receptor_complex", "interaction")


type CurveTransform = Callable[[NDArray[np.floating]], NDArray[np.floating]]
"""Rescales a ``g(r)`` curve before curves are compared (default: :func:`_log2_floor`)."""


def _log2_floor(g: NDArray[np.floating]) -> NDArray[np.floating]:
    """log2 of g floored at 0.05, so empty/depleted bins stay finite (~-4.3)."""
    # `np.asarray` only to keep it typed: numpy's ufunc stubs widen to `Any`
    # It is a no-op on an array.
    return np.asarray(np.log2(np.maximum(g, 0.05)))


@d.dedent
def get_lric_auc(
    adata: AnnData | None = None,
    uns_key: str = "lric",
    liana_res: pd.DataFrame | None = None,
    max_dist: float | None = None,
    transform_fn: CurveTransform = _log2_floor,
    min_bins: int = 3,
) -> pd.DataFrame:
    """
    Summarise a ``lric`` / ``cross_pcf`` result into one score per interaction.

    Each interaction's ``g(r)`` profile is reduced to the span-normalised area
    under its ``log2 g(r)`` curve -- a mean log2 fold change: ``> 0``
    co-enriched, ``< 0`` depleted, ``0`` random.

    Parameters
    ----------
    %(adata)s
        Its ``.uns[uns_key]`` holds the result. Mutually exclusive with
        ``liana_res``.
    %(uns_key)s
    %(liana_res)s
        A ``lric`` / ``cross_pcf`` result, used when ``adata`` is ``None``.
    max_dist
        Integrate only over radii ``r < max_dist``; ``None`` uses all radii.
    transform_fn
        Applied to ``g`` before integrating; defaults to log2 with ``g``
        floored at ``0.05``, so empty/depleted bins stay finite and count as
        strong depletion. Pass :obj:`numpy.log2` for the strict behaviour
        where non-finite values (e.g. ``log2(0) = -inf``) are dropped from
        the integral.
    min_bins
        Drop interactions with fewer than this many finite bins in the window
        (a support gate for proportion-masked / degenerate interactions).

    Returns
    -------
    A ``pandas.DataFrame`` with the id columns of the input (whichever of
    ``source``, ``target``, ``ligand_complex``, ``receptor_complex``,
    ``interaction`` are present) plus ``score`` and ``peak_radius`` -- the
    radius at which ``|transform_fn(g)|`` is largest, i.e. where the
    interaction deviates most from the null -- sorted most-enriched first.
    The column names match :func:`liana.pl.dotplot`'s expectations. Empty if
    nothing clears ``min_bins``.

    Examples
    --------
    Rank the interactions of a spatial result by how co-enriched they are across radius -- here the cell-type-agnostic LRIC of :func:`liana.mt.lric.__call__`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> li.mt.lric(adata, resource_name="consensus", key_added="lric")
    >>> scores = li.mt.get_lric_auc(adata, uns_key="lric")
    """
    res = _get_liana_res(adata, liana_res, uns_key)

    ids = [c for c in _ID_COLS if c in res.columns]
    if not ids:
        raise ValueError(f"None of {_ID_COLS} found in the result's columns.")

    # (n_groups, n_radii) wide matrix -- the radius grid is shared by every group
    radii = np.unique(res["radius"].to_numpy(dtype=float))
    gid, uniques = pd.MultiIndex.from_frame(res[ids]).factorize()
    if not isinstance(uniques, pd.MultiIndex):  # `factorize` of a MultiIndex yields one
        raise TypeError(f"expected a MultiIndex, got {type(uniques).__name__}")
    keys = uniques.set_names(ids)  # `factorize` drops the level names
    rid = np.searchsorted(radii, res["radius"].to_numpy(dtype=float))
    Y = np.full((len(keys), radii.size), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        Y[gid, rid] = transform_fn(res["g"].to_numpy(dtype=float))

    keep = np.isfinite(Y)
    if max_dist is not None:
        keep &= radii < max_dist
    # mask-aware trapezoid: a segment contributes only if both its ends are kept
    seg = keep[:, :-1] & keep[:, 1:]
    Yv = np.where(keep, Y, 0.0)
    area = (0.5 * (Yv[:, :-1] + Yv[:, 1:]) * np.diff(radii) * seg).sum(axis=1)
    # radii are ascending, so the kept span is between the first and last kept bin
    lo, hi = radii[np.argmax(keep, axis=1)], radii[::-1][np.argmax(keep[:, ::-1], axis=1)]
    span = np.where(keep.any(axis=1), hi - lo, 0.0)

    ok = (keep.sum(axis=1) >= min_bins) & (span > 0)
    if not ok.any():
        in_window = radii.size if max_dist is None else int((radii < max_dist).sum())
        if in_window < min_bins:
            msg = (
                f"only {in_window} radius bin(s) lie within max_dist={max_dist}, below "
                f"min_bins={min_bins} — set min_bins<={in_window}, raise max_dist, "
                "or recompute with a finer radius_step"
            )
        else:
            msg = (
                f"every interaction has <{min_bins} finite g(r) bins in-window "
                "(nz_prop/expr_prop masking / sparse geometry) — lower min_bins or the threshold"
            )
        _logg(msg, level="warn", verbose=True)
    out = keys[ok].to_frame(index=False)
    out["score"] = area[ok] / span[ok]
    out["peak_radius"] = radii[np.where(keep, np.abs(Y), -np.inf).argmax(axis=1)][ok]
    return out.sort_values("score", ascending=False, ignore_index=True)


@d.dedent
def get_lric_divergence(
    adata: AnnData | None = None,
    uns_key: str = "lric",
    liana_res: pd.DataFrame | None = None,
    feature_a: dict[str, object] | None = None,
    feature_b: dict[str, object] | None = None,
    max_dist: float | None = None,
    transform_fn: CurveTransform = _log2_floor,
    min_bins: int = 3,
) -> pd.Series:
    """
    Compare the full ``g(r)`` profiles of two curves from ``lric`` / ``cross_pcf``.

    Where :func:`get_lric_auc` collapses each curve to a signed mean (so opposite
    deviations at different radii cancel), the divergence is the span-normalised
    area *between* two ``transform_fn(g(r))`` curves -- ``0`` means identical
    spatial profiles, larger means more different -- and reports where along
    radius the separation peaks.

    Parameters
    ----------
    %(adata)s
        Its ``.uns[uns_key]`` holds the result. Mutually exclusive with
        ``liana_res``.
    %(uns_key)s
    %(liana_res)s
        A ``lric`` / ``cross_pcf`` result, used when ``adata`` is ``None``.
        May be a concatenation of several results with extra annotation columns
        (e.g. ``condition``) -- pin those in the selections to compare the same
        interaction across conditions.
    feature_a
        Selection of the first curve as ``{column: value}`` over any columns of
        the result, e.g. ``dict(interaction="Dcn^Egfr")`` or
        ``dict(interaction="Dcn^Egfr", condition="stim")``. It must resolve to a
        single interaction; rows it leaves unpinned (e.g. replicate samples)
        average into one curve per radius.
    feature_b
        Selection of the second curve; same rules as ``feature_a``.
    max_dist
        Compare only over radii ``r < max_dist``; ``None`` uses all radii.
    transform_fn
        Applied to ``g`` before comparing; defaults to log2 with ``g``
        floored at ``0.05``, so empty/depleted bins stay finite. Pass
        :obj:`numpy.log2` for the strict behaviour where radii with a
        non-finite transformed curve are dropped from the comparison.
    min_bins
        Minimum shared finite radius bins required; fewer raises a ``ValueError``.

    Returns
    -------
    A ``pandas.Series`` with ``divergence`` (mean ``|A - B|`` across radius),
    ``r_star`` (radius of the largest separation), ``delta_star`` (signed ``A - B``
    there), ``direction``, the selections and their display labels.

    Examples
    --------
    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> li.mt.cross_pcf(adata, groupby="bulk_labels", key_added="cross_pcf")
    >>> div = li.mt.get_lric_divergence(
    ...     adata,
    ...     "cross_pcf",
    ...     feature_a=dict(source="CD14+ Monocyte", target="CD34+"),
    ...     feature_b=dict(source="CD14+ Monocyte", target="CD19+ B"),
    ... )
    """
    res = _get_liana_res(adata, liana_res, uns_key)
    if not feature_a or not feature_b:
        raise ValueError("`feature_a` and `feature_b` selections must be provided!")

    ids = [c for c in _ID_COLS if c in res.columns]
    a = _mean_curve(res, feature_a, ids, transform_fn)
    b = _mean_curve(res, feature_b, ids, transform_fn)

    delta = (a - b).replace([np.inf, -np.inf], np.nan).dropna()
    if max_dist is not None:
        delta = delta[delta.index < max_dist]
    if len(delta) < max(min_bins, 2):
        raise ValueError(f"Only {len(delta)} shared finite bins; `min_bins` is {min_bins}.")

    x, dv = delta.index.to_numpy(), delta.to_numpy()
    j = np.argmax(np.abs(dv))
    return pd.Series(
        {
            "label_a": " | ".join(map(str, feature_a.values())),
            "label_b": " | ".join(map(str, feature_b.values())),
            "divergence": np.trapezoid(np.abs(dv), x) / (x[-1] - x[0]),
            "r_star": x[j],
            "delta_star": dv[j],
            "direction": "A > B" if dv[j] > 0 else "B > A" if dv[j] < 0 else "equal",
            "feature_a": feature_a,
            "feature_b": feature_b,
            "max_dist": max_dist,
        }
    )


def _mean_curve(
    res: pd.DataFrame,
    sel: dict[str, object],
    ids: list[str],
    transform_fn: CurveTransform,
) -> pd.Series:
    """One `transform_fn(g)` curve indexed by radius; replicate rows average."""
    mask = np.ones(len(res), dtype=bool)
    for col, value in sel.items():
        if col not in res.columns:
            raise ValueError(f"`{col}` is not a column of this result; it has {list(res.columns)}.")
        mask &= (res[col] == value).to_numpy()
        if not mask.any():
            raise ValueError(f"No rows match {sel}.")
    sub = res.loc[mask]
    loose = [c for c in ids if c not in sel and sub[c].nunique() > 1]
    if loose:
        raise ValueError(f"{sel} matches more than one interaction -- also pin {loose}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        y = transform_fn(sub["g"].to_numpy(dtype=float))
    return pd.Series(y, index=sub["radius"].to_numpy(dtype=float)).groupby(level=0).mean()
