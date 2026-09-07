from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import norm
from tqdm import tqdm

from liana._core._common import _logg
from liana.method.sp._utils import _spatialdm_weight_norm, _zscore

type Weight = np.ndarray | csr_matrix
"""A spatial connectivity matrix; `_handle_connectivity` yields the sparse form."""


class GlobalStat(Protocol):
    """A global bivariate statistic over `x`, `y` and a connectivity weight."""

    def __call__(self, x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray: ...


class GlobalFunction:
    """
    Metaclass for wrapping global functions

    Parameters
    ----------
    fun
        The function itself
    name
        The name of the function

    Attributes
    ----------
    fun
        The function itself
    name
        The name of the function
    pvals_name
        The name of the function with `'_pvals'` appended
    """

    instances: dict[str, GlobalFunction] = {}

    def __init__(self, fun: GlobalStat, name: str) -> None:
        self.fun = fun
        self.name = name
        self.pvals_name = self.name + "_pvals"

        GlobalFunction.instances[name] = self

    def _permutation_pvals(
        self,
        x_mat: np.ndarray,
        y_mat: np.ndarray,
        weight: Weight,
        global_stat: np.ndarray,
        n_perms: int,
        mask_negatives: bool,
        seed: int,
        verbose: bool,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        # initialize mat /w n_perms * number of X->Y
        idx = x_mat.shape[0]

        # permutation mat /w n_permss x LR_n
        perm_mat = np.zeros((n_perms, global_stat.shape[0]))

        for perm in tqdm(range(n_perms), disable=not verbose):
            _idx = rng.permutation(idx)
            perm_mat[perm, :] = self.fun(x_mat=x_mat[_idx, :], y_mat=y_mat[_idx, :], weight=weight)

        if mask_negatives:
            global_pvals = 1 - (global_stat > perm_mat).sum(axis=0) / n_perms
        else:
            global_pvals = 1 - (np.abs(global_stat) > np.abs(perm_mat)).sum(axis=0) / n_perms

        return np.asarray(global_pvals)

    def _zscore_pvals(
        self,
        weight: Weight,
        global_stat: np.ndarray,
        mask_negatives: bool,
    ) -> np.ndarray:
        """SpatialDM's global z-score p-value calculation"""
        dense = weight if isinstance(weight, np.ndarray) else np.asarray(weight.todense())
        spot_n = dense.shape[0]

        # global distance/weight variance as in spatialDM
        numerator = spot_n**2 * ((dense * dense).sum()) - (2 * spot_n * (dense @ dense).sum()) + (dense.sum() ** 2)
        denominator = spot_n**2 * (spot_n - 1) ** 2
        weight_var_sq = (numerator / denominator) ** (1 / 2)

        global_zscores = global_stat / weight_var_sq

        if mask_negatives:
            global_zpvals = norm.sf(global_zscores)
        else:
            global_zpvals = norm.sf(np.abs(global_zscores)) * 2

        return np.asarray(global_zpvals)

    def __call__(
        self,
        xy_stats: pd.DataFrame,
        x_mat: np.ndarray | csr_matrix,
        y_mat: np.ndarray | csr_matrix,
        weight: Weight,
        seed: int,
        n_perms: int | None,
        mask_negatives: bool,
        verbose: bool,
    ) -> None:
        """
        Function caller wrapper

        Parameters
        ----------
        xy_stats
            Dictionary where stats and p-values are stored
        x_mat
            2D array with x variables
        y_mat
            2D array with y variables
        weight
            Connectivity weight matrix
        %(seed)s
        %(n_perms)s
        %(mask_negatives)s
        %(verbose)s

        Raises
        ------
        ValueError
            If the given function is not supported
        """
        norm_weight: Weight
        x_dense: np.ndarray
        y_dense: np.ndarray
        if self.name == "morans":
            x_dense = _zscore(x_mat, axis=0, global_r=True)
            y_dense = _zscore(y_mat, axis=0, global_r=True)
            norm_weight = _spatialdm_weight_norm(weight)
        elif self.name == "lee":
            x_dense = _zscore(x_mat)
            y_dense = _zscore(y_mat)
            norm_weight = weight * weight
        else:
            raise ValueError("Global function not supported")

        global_stat = self.fun(x_mat=x_dense, y_mat=y_dense, weight=norm_weight)

        global_pvals: np.ndarray | None = None
        if n_perms is None:
            global_pvals = None
        elif n_perms > 0:
            global_pvals = self._permutation_pvals(
                x_mat=x_dense,
                y_mat=y_dense,
                weight=norm_weight,
                global_stat=global_stat,
                n_perms=n_perms,
                mask_negatives=mask_negatives,
                seed=seed,
                verbose=verbose,
            )
        elif n_perms == 0 and self.name == "morans":
            global_pvals = self._zscore_pvals(
                weight=norm_weight, global_stat=global_stat, mask_negatives=mask_negatives
            )
        elif n_perms == 0 and self.name == "lee":
            _logg("Global Lee does not support analytical p-values", "warning", verbose=verbose)

        xy_stats[self.name] = global_stat
        xy_stats[self.pvals_name] = global_pvals


def _morans_stat(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    return np.asarray(((weight @ x_mat) * y_mat).sum(axis=0))


def _lee_stat(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    return np.asarray(((weight @ x_mat) * y_mat).sum(axis=0) / weight.sum())


_global_r = GlobalFunction(_morans_stat, "morans")
_global_l = GlobalFunction(_lee_stat, "lee")
