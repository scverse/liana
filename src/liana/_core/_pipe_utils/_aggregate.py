from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata

from liana._core._common import _logg

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from liana.method.sc._rank_aggregate import AggregateClass


def _aggregate(
    lrs: dict[str, pd.DataFrame],
    consensus: AggregateClass,
    aggregate_method: Literal["rra", "mean"] = "rra",
    _consensus_opts: list[str] | None = None,
    _key_cols: list[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Function to aggregate the results of all methods into a single DataFrame.

    Parameters
    ----------
    lrs
        a list with results for all methods
    consensus
        ConsensusClass instance used to generate the lr results
    _key_cols
        should represent unique LRs columns by which to join
    aggregate_method
        method by which we aggregate the ranks. Options are ['rra', 'mean'],
        where 'rra' corresponds to the RRA method;
        while 'mean' is just the mean of the ranks divided by the number of interactions
    _consensus_opts
        consensus ranks to be obtained
    verbose
        Verbosity flag

    Returns
    -------
    A long pd.DataFrame with ranked LRs
    """
    # join the sc to the whole universe between the methods
    if _key_cols is None:
        _key_cols = ["source", "target", "ligand_complex", "receptor_complex"]
    if _consensus_opts is None:
        _consensus_opts = ["Magnitude", "Specificity"]

    frames = [lrs[method].drop_duplicates(keep="first") for method in lrs]
    # reduce to a df with the shared keys + all relevant sc
    lr_res = reduce(
        lambda left, right: pd.merge(left, right, how="outer", on=_key_cols, suffixes=("", "_duplicated")), frames
    )
    # drop duplicated columns
    lr_res = lr_res.loc[:, ~lr_res.columns.str.endswith("_duplicated")]

    order_col = ""
    if "Specificity" in _consensus_opts:
        if consensus.specificity is None:
            raise ValueError("Cannot aggregate specificity ranks: `consensus.specificity` is unset.")
        lr_res[consensus.specificity] = _rank_aggregate(
            lr_res, consensus.specificity_specs, aggregate_method=aggregate_method, verbose=verbose
        )
        order_col = consensus.specificity
    if "Magnitude" in _consensus_opts:
        if consensus.magnitude is None:
            raise ValueError("Cannot aggregate magnitude ranks: `consensus.magnitude` is unset.")
        lr_res[consensus.magnitude] = _rank_aggregate(
            lr_res, consensus.magnitude_specs, aggregate_method=aggregate_method, verbose=verbose
        )
        order_col = consensus.magnitude

    lr_res = lr_res.sort_values(order_col)

    return lr_res


def _rank_aggregate(
    lr_res: pd.DataFrame,
    specs: dict[str, tuple[str, bool | None]],
    aggregate_method: Literal["rra", "mean"],
    verbose: bool = False,
) -> NDArray[np.floating]:
    """
    Aggregate method ranks

    Parameters
    ----------
    lr_res
        joined results from all methods
    specs
        specs dictionary where method_name:(score_name, score_desc)
    aggregate_method
        method by which to aggregate the ranks
    verbose
        Verbosity flag

    Returns
    -------
    An array of values /w length of lr_res.shape[0]
    """
    if aggregate_method not in ("rra", "mean"):
        raise ValueError(f"`aggregate_method` must be 'rra' or 'mean', got {aggregate_method!r}.")

    # methods whose score was not computed (e.g. permutation p-values with `n_perms=None`) have no column
    specs = {method: spec for method, spec in specs.items() if spec[0] in lr_res.columns}
    # rank each unique score column once (Connectome and NATMI share `expr_prod`)
    columns: dict[str, bool | None] = {}
    for col, asc in specs.values():
        if columns.setdefault(col, asc) != asc:
            raise ValueError(f"Column `{col}` is ranked in opposite directions by different methods.")
    if len(columns) < 2:
        _logg(f"Aggregating over {len(columns)} score(s) only: {sorted(columns)}.", level="warn", verbose=verbose)

    rmat: NDArray[np.floating] = np.column_stack(
        [rankdata(lr_res[col] if asc else -lr_res[col], method="average") for col, asc in columns.items()]
    )

    if aggregate_method == "rra":
        return _robust_rank_aggregate(rmat)
    return np.asarray(np.mean(rmat, axis=1) / rmat.shape[0], dtype=np.float64)


def _corr_beta_pvals(p: NDArray[np.floating], k: int) -> NDArray[np.floating]:
    """
    Correct beta p-values

    Parameters
    ----------
    p
        (min) p-value
    k
        total number of rows

    Returns
    -------
    An array with corrected p-values
    """
    p = np.clip(p * k, a_min=0, a_max=1)
    return p


def _rho_scores(
    rmat: NDArray[np.floating],
    dist_a: NDArray[np.integer],
    dist_b: NDArray[np.integer],
) -> NDArray[np.floating]:
    """
    Calculate Beta Distribution Rho Scores

    Parameters
    ----------
    rmat
        a matrix where rows are the ranks/n for each interaction, while
        columns correspond to each method
    dist_a
        non-negative shape param a
    dist_b
        non-negative shape param b

    Returns
    -------
    A vector of pvals as implemented in the RRA method
    """
    # Sort values by sources (rows)
    rmat = np.sort(rmat, axis=1)
    # Calc beta cdf across rows
    p = beta.cdf(rmat, dist_a, dist_b)
    # get min pval per row
    p = np.min(p, axis=1)
    # correct p-vals
    rho = _corr_beta_pvals(p, k=rmat.shape[1])

    return rho


def _robust_rank_aggregate(rmat: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Calculate Robust Rank Aggregate as in Kolde et al., 2012

    Parameters
    ----------
    rmat
        Matrix with interaction ranks (rows) for each method (columns)

    Returns
    -------
    An array with p-values for each row
    """
    # 0-1 values depending on relative rank of
    # each interaction divided by the max of each method
    # due to max diffs due to ties
    rmat = rmat / np.max(rmat, axis=0)
    # generate dist_a/b with same row size as rmat
    dist_a = np.repeat([np.arange(rmat.shape[1])], rmat.shape[0], axis=0) + 1
    dist_b = rmat.shape[1] - dist_a + 1

    return _rho_scores(rmat, dist_a, dist_b)
