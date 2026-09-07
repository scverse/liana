from collections.abc import Callable

import numpy as np
from pandas import DataFrame

from liana._core._constants import DefaultValues as V
from liana._core._docs import d
from liana._core._types import ScoreTransform
from liana.method.df_to_lr import df_to_lr
from liana.method.fun._causalnet import find_causalnet
from liana.method.fun._estimate_metalinks import estimate_metalinks
from liana.method.sc import (
    cellchat,
    cellphonedb,
    connectome,
    geometric_mean,
    logfc,
    natmi,
    scseqcomm,
    singlecellsignalr,
)
from liana.method.sc._Method import Method as _Method
from liana.method.sc._Method import MethodMeta as _MethodMeta
from liana.method.sc._Method import _show_methods
from liana.method.sc._rank_aggregate import AggregateClass
from liana.method.sc._rank_aggregate import _rank_aggregate_meta as aggregate_meta
from liana.method.sp import (
    MistyData,
    bivariate,
    compute_global_specificity,
    cross_pcf,
    genericMistyData,
    inflow,
    lric,
    lrMistyData,
)
from liana.method.sp._lric_helpers import get_lric_auc, get_lric_divergence

# callable consensus instance
_methods = [cellphonedb, connectome, logfc, natmi, singlecellsignalr]
rank_aggregate = AggregateClass(aggregate_meta, methods=_methods)

_all_methods: list[_Method | AggregateClass] = [*_methods, rank_aggregate, geometric_mean, scseqcomm, cellchat]
"""Every method liana ships, in the order :func:`show_methods` lists them."""


def show_methods() -> DataFrame:
    """
    Shows methods available in LIANA

    Returns
    -------
    Table of methods available.

    Examples
    --------
    One row per method, with the scores it reports and its reference. Each name
    corresponds to a callable instance in `liana.method`, e.g.
    ``liana.mt.cellphonedb``:

    >>> import liana as li
    >>> methods = li.mt.show_methods()
    """
    return _show_methods(_all_methods)


def get_method_scores() -> dict[str, bool | None]:
    """
    Shows the scoring methods available.

    Returns
    -------
    Dictionary of all scoring functions, with a boolean indicating whether the score is ascending or not

    Examples
    --------
    Maps every score that liana's methods report to whether a *lower* value is the
    stronger one -- `True` for p-values and for the aggregate ranks.
    :func:`liana.mt.process_scores` uses this to decide what to invert:

    >>> import liana as li
    >>> scores = li.mt.get_method_scores()
    """
    specificity_scores = {
        method.specificity: method.specificity_ascending for method in _all_methods if method.specificity is not None
    }
    magnitude_scores = {
        method.magnitude: method.magnitude_ascending for method in _all_methods if method.magnitude is not None
    }

    scores = {**specificity_scores, **magnitude_scores}
    return scores


@d.dedent
def process_scores(liana_res: DataFrame, score_key: str, inverse_fn: ScoreTransform = V.inverse_fn) -> DataFrame:
    """
    Processes and outputs a given score.

    Parameters
    ----------
    %(liana_res)s
    %(score_key)s
    %(inverse_fn)s

    Returns
    -------
    A `DataFrame` with the processed scores.

    Examples
    --------
    `magnitude_rank` is one of the scores for which a lower value is the stronger
    one, so it is inverted with `inverse_fn`. A score that is already
    "higher is stronger" is returned unchanged:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> li.mt.rank_aggregate(adata, groupby="bulk_labels", n_perms=None)
    >>> res = li.mt.process_scores(adata.uns["liana_res"], score_key="magnitude_rank")
    """
    df = liana_res.copy()
    scores = get_method_scores()

    if not np.isin(score_key, list(scores.keys())).any():
        raise ValueError(f"Score column {score_key} not found in liana's method scores.")

    # reverse if ascending order
    ascending_order = scores[score_key]
    if ascending_order:
        df[score_key] = inverse_fn(df[score_key])

    return df


__all__ = [
    "MistyData",
    "bivariate",
    "cellchat",
    "cellphonedb",
    "compute_global_specificity",
    "connectome",
    "cross_pcf",
    "df_to_lr",
    "estimate_metalinks",
    "find_causalnet",
    "genericMistyData",
    "geometric_mean",
    "get_lric_auc",
    "get_lric_divergence",
    "get_method_scores",
    "inflow",
    "logfc",
    "lrMistyData",
    "lric",
    "natmi",
    "process_scores",
    "rank_aggregate",
    "scseqcomm",
    "show_methods",
    "singlecellsignalr",
]

# Remove these
del Callable, DataFrame, ScoreTransform, V, d
