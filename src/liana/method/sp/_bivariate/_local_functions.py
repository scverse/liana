from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.stats import norm, rankdata
from tqdm import tqdm

from liana.method.sp._bivariate._global_functions import Weight

if TYPE_CHECKING:
    # `prange.__new__` returns a `range` but is unannotated
    # numba resolves the alias through the globals, so `parallel=True` still works
    prange = range
else:
    prange = nb.prange
from liana.method.sp._utils import _spatialdm_weight_norm, _zscore

type LocalStat = Callable[..., np.ndarray]
"""A local bivariate statistic over `x`, `y` and a connectivity weight.

Spelled with `...` parameters because some of these are `numba.njit` dispatchers,
which no explicit signature matches structurally.
"""


class LocalFunction:
    """
    Class representing information about bivariate spatial functions.

    Parameters
    ----------
    name
        Name of the function
    metadata
        Short description for the function
    fun
        The actual function
    reference
        Reference/description for the function

    Attributes
    ----------
    name
        Name of the function
    metadata
        Short description for the function
    fun
        The actual function
    reference
        Reference/description for the function
    """

    instances: dict[str, LocalFunction] = {}

    def __init__(
        self,
        name: str,
        metadata: str,
        fun: LocalStat,
        reference: str | None = None,
    ) -> None:
        self.name = name
        self.metadata = metadata
        self.fun = fun
        self.reference = reference

        LocalFunction.instances[name] = self

    def __call__(
        self,
        x_mat: np.ndarray | csr_matrix,
        y_mat: np.ndarray | csr_matrix,
        weight: Weight,
        n_perms: int | None,
        seed: int,
        mask_negatives: bool,
        verbose: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Function caller wrapper

        Parameters
        ----------
        x_mat
            2D array with x variables
        y_mat
            2D array with y variables
        weight
            Connectivity weight matrix
        %(n_perms)s
        %(seed)s
        %(mask_negatives)s
        %(verbose)s

        Returns
        -------
        local_scores
            Matrix with the local scores
        local_pvals
            Matrix of resulting p-values
        """
        norm_weight: Weight = weight
        x_dense: np.ndarray
        y_dense: np.ndarray
        if self.name == "morans":
            x_dense = self._norm_max(x_mat)
            y_dense = self._norm_max(y_mat)
            norm_weight = _spatialdm_weight_norm(weight)
        else:
            x_dense = x_mat.toarray() if isinstance(x_mat, csr_matrix) else x_mat
            y_dense = y_mat.toarray() if isinstance(y_mat, csr_matrix) else y_mat

        if ("masked" in self.name or norm_weight.shape[0] < 10000) and isinstance(norm_weight, csr_matrix):
            norm_weight = np.asarray(norm_weight.todense())

        local_scores = self.fun(x_dense, y_dense, norm_weight)

        local_pvals: np.ndarray | None = None
        if n_perms is None:
            local_pvals = None
        elif n_perms > 0:
            local_pvals = self._permutation_pvals(
                x_mat=x_dense,
                y_mat=y_dense,
                weight=norm_weight,
                local_truth=local_scores,
                n_perms=n_perms,
                seed=seed,
                mask_negatives=mask_negatives,
                verbose=verbose,
            )
        elif n_perms == 0:
            local_pvals = self._zscore_pvals(
                x_mat=x_dense,
                y_mat=y_dense,
                weight=norm_weight,
                local_truth=local_scores,
                mask_negatives=mask_negatives,
            )

        return local_scores, local_pvals

    def __repr__(self) -> str:
        return f"{self.name}: {self.metadata}"

    def _permutation_pvals(
        self,
        x_mat: np.ndarray,
        y_mat: np.ndarray,
        weight: Weight,
        local_truth: np.ndarray,
        n_perms: int,
        seed: int,
        mask_negatives: bool,
        verbose: bool,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        spot_n = local_truth.shape[0]
        xy_n = local_truth.shape[1]

        local_pvals = np.zeros((spot_n, xy_n))

        # shuffle the matrix
        for _ in tqdm(range(n_perms), disable=not verbose):
            _idx = rng.permutation(spot_n)
            perm_score = self.fun(x_mat=x_mat[_idx, :], y_mat=y_mat[_idx, :], weight=weight)
            if mask_negatives:
                local_pvals += np.array(perm_score >= local_truth, dtype=int)
            else:
                local_pvals += np.array(np.abs(perm_score) >= np.abs(local_truth), dtype=int)

        return np.asarray(local_pvals / n_perms)

    def _zscore_pvals(
        self,
        x_mat: np.ndarray,
        y_mat: np.ndarray,
        local_truth: np.ndarray,
        weight: Weight,
        mask_negatives: bool,
    ) -> np.ndarray:
        """
        Local Moran's R analytical p-values as in spatialDM (Li et al., 2022)

        Parameters
        ----------
        x_mat
            2D array with x variables
        y_mat
            2D array with y variables
        local_truth
            2D array with Local Moran's I
        weight
            Connectivity weights
        mask_negatives
            Whether to mask negative correlations pvalue

        Returns
        -------
        2D array of p-values with shape(n_spot, xy_n)
        """
        spot_n = x_mat.shape[0]

        x_norm = np.apply_along_axis(norm.fit, axis=0, arr=x_mat)
        y_norm = np.apply_along_axis(norm.fit, axis=0, arr=y_mat)

        # get x,y std
        x_sigma, y_sigma = x_norm[1, :], y_norm[1, :]

        x_sigma = x_sigma * spot_n / (spot_n - 1)
        y_sigma = y_sigma * spot_n / (spot_n - 1)

        std = self._get_local_var(x_sigma, y_sigma, weight, spot_n)
        local_zscores = local_truth / std

        if mask_negatives:
            local_zpvals = norm.sf(local_zscores)
        else:
            local_zpvals = norm.sf(np.abs(local_zscores))

        return np.asarray(local_zpvals)

    def _get_local_var(
        self,
        x_sigma: np.ndarray,
        y_sigma: np.ndarray,
        weight: Weight,
        spot_n: int,
    ) -> np.ndarray:
        """
        Spatial weight variance as in spatialDM (Li et al., 2022)

        Parameters
        ----------
        x_sigma
            Standard deviations for each x (e.g. std of all ligands)
        y_sigma
            Standard deviations for each y (e.g. std of all receptors)
        weight
            Connectivity weight matrix
        spot_n
            number of spots/cells in the matrix

        Returns
        -------
        2D array of standard deviations with shape(n_spot, xy_n)
        """
        dense = weight if isinstance(weight, np.ndarray) else np.asarray(weight.todense())

        weight_sq = (dense**2).sum(axis=1)

        dim = 2 * (spot_n - 1) ** 2 / spot_n**2
        sigma_prod = x_sigma * y_sigma
        core = dim * sigma_prod

        var = np.multiply.outer(weight_sq, core) + core

        return np.asarray(var**0.5)

    def _norm_max(self, X: np.ndarray | csr_matrix, axis: int = 0) -> np.ndarray:
        maxima = X.max(axis=axis)
        dense_max = maxima.toarray() if isinstance(maxima, csr_matrix | coo_matrix) else maxima
        zscored = _zscore(X / dense_max, axis=axis)

        return np.where(np.isnan(zscored), 0, zscored)

    @classmethod
    def _get_instance(cls, name: str) -> LocalFunction:
        name = name.lower()
        instances = cls.instances
        if name not in instances:
            raise ValueError(f"Function {name} not found. Available functions are: {', '.join(instances)}")

        return cls.instances[name]


@nb.njit(nb.float32(nb.float32[:], nb.float32[:], nb.float32[:], nb.float32), cache=True)
def _wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray, wsum: float) -> float:

    x = np.argsort(x).argsort().astype(np.float32)
    y = np.argsort(y).argsort().astype(np.float32)

    wx = w * x
    wy = w * y

    numerator = wsum * sum(wx * y) - sum(wx) * sum(wy)

    denominator_x = wsum * sum(w * (x**2)) - sum(wx) ** 2
    denominator_y = wsum * sum(w * (y**2)) - sum(wy) ** 2
    denominator = denominator_x * denominator_y

    if (denominator == 0) or (numerator == 0):
        return 0.0

    corr: float = numerator / (denominator**0.5)
    return corr


@nb.njit(nb.float32[:, :](nb.float32[:, :], nb.float32[:, :], nb.float32[:, :]), parallel=True, cache=True)
def _masked_spearman(x_mat: np.ndarray, y_mat: np.ndarray, weight: np.ndarray) -> np.ndarray:
    spot_n = x_mat.shape[0]
    xy_n = x_mat.shape[1]

    local_corrs = np.zeros((spot_n, xy_n), dtype=np.float32)

    for i in prange(spot_n):
        w = weight[i, :]
        msk = w > 0
        wsum = sum(w[msk])

        for j in range(xy_n):
            x = x_mat[:, j][msk]
            y = y_mat[:, j][msk]

            local_corrs[i, j] = _wcorr(x, y, w[msk], wsum)

    # NOTE done due to numpy/numba sum imprecision, https://github.com/numba/numba/issues/8749
    local_corrs = np.clip(a=local_corrs, a_min=-1.0, a_max=1.0, out=local_corrs)

    return local_corrs


def _vectorized_correlations(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    weight: Weight,
    method: str = "pearson",
) -> np.ndarray:
    """
    Vectorized implementation of weighted correlations.

    Note: due to the imprecision of np.sum and np.dot, the function is accurate to 5 decimal places.
    """
    if method not in ["pearson", "spearman"]:
        raise ValueError("method must be one of 'pearson', 'spearman'")
    weight_sums = np.asarray(weight.sum(axis=1)).reshape(-1, 1)

    if method == "spearman":
        x_mat = rankdata(x_mat, axis=0)
        y_mat = rankdata(y_mat, axis=0)

    # standard pearson
    n1 = weight_sums * (weight @ (x_mat * y_mat))
    n2 = (weight @ x_mat) * (weight @ y_mat)
    numerator = n1 - n2

    ss_x = weight_sums * (weight @ x_mat**2)
    ss_y = weight_sums * (weight @ y_mat**2)
    denominator_x = ss_x - (weight @ x_mat) ** 2
    denominator_y = ss_y - (weight @ y_mat) ** 2

    # dealt with instability under 6th decimal place
    denominator_x[denominator_x <= 1e-6 * ss_x] = 0
    denominator_y[denominator_y <= 1e-6 * ss_y] = 0
    denominator = denominator_x * denominator_y
    denominator = denominator**0.5

    zeros = np.zeros(numerator.shape)
    local_corrs = np.divide(numerator, denominator, out=zeros, where=denominator != 0)

    # NOTE done due to numpy/numba sum imprecision, https://github.com/numba/numba/issues/8749
    local_corrs = np.clip(local_corrs, -1, 1, out=local_corrs, dtype=np.float32)

    return np.asarray(local_corrs)


def _vectorized_pearson(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    return _vectorized_correlations(x_mat, y_mat, weight, method="pearson")


def _vectorized_spearman(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    return _vectorized_correlations(x_mat, y_mat, weight, method="spearman")


def _vectorized_cosine(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    xy_dot = weight @ (x_mat * y_mat)
    x_dot = weight @ (x_mat**2)
    y_dot = weight @ (y_mat**2)
    denominator = (x_dot * y_dot) + np.finfo(np.float32).eps

    return np.asarray(xy_dot / denominator**0.5)


def _vectorized_jaccard(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    x_mat, y_mat = x_mat > 0, y_mat > 0  ## NOTE only positive
    numerator = weight @ np.minimum(x_mat, y_mat)
    denominator = weight @ np.maximum(x_mat, y_mat) + np.finfo(np.float32).eps

    return numerator / denominator


def _local_morans(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    """
    Local Moran's I

    Parameters
    ----------
    x_mat
        2D array with x variables
    y_mat
        2D array with y variables
    weight
        Connectivity weight matrix


    Returns
    -------
    Returns 2D array of local Moran's I with shape(n_spot, xy_n)
    """
    local_x = x_mat * (weight @ y_mat)
    local_y = y_mat * (weight @ x_mat)
    return np.asarray(local_x + local_y)


def _product(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    return np.asarray((weight @ x_mat) * (weight @ y_mat))


def _norm_product(x_mat: np.ndarray, y_mat: np.ndarray, weight: Weight) -> np.ndarray:
    x_mat = weight @ x_mat
    y_mat = weight @ y_mat

    x_norm = np.max(np.abs(x_mat), axis=0)
    y_norm = np.max(np.abs(y_mat), axis=0)

    x_norm[x_norm == 0.0] = 1.0
    y_norm[y_norm == 0.0] = 1.0

    return np.asarray((x_mat / x_norm) * (y_mat / y_norm))


_bivariate_functions = [
    LocalFunction(
        name="pearson",
        metadata="weighted Pearson correlation coefficient",
        fun=_vectorized_pearson,
    ),
    LocalFunction(
        name="spearman",
        metadata="weighted Spearman correlation coefficient",
        fun=_vectorized_spearman,
    ),
    LocalFunction(
        name="cosine",
        metadata="weighted Cosine similarity",
        fun=_vectorized_cosine,
    ),
    LocalFunction(
        name="jaccard",
        metadata="weighted Jaccard similarity",
        fun=_vectorized_jaccard,
    ),
    LocalFunction(
        name="product",
        metadata="simple weighted product",
        fun=_product,
        reference="If vars are z-scaled = Lee's static (Lee 2021;J.Geograph.Syst.)",
    ),
    LocalFunction(
        name="norm_product",
        metadata="normalized weighted product",
        fun=_norm_product,
    ),
    LocalFunction(
        name="morans",
        metadata="Moran's R",
        fun=_local_morans,
        reference="Li, Z., Wang, T., Liu, P. and Huang, Y., 2022. SpatialDM:"
        "Rapid identification of spatially co-expressed ligand-receptor"
        "reveals cell-cell communication patterns. bioRxiv, pp.2022-08.",
    ),
    LocalFunction(
        name="masked_spearman",
        metadata="masked & weighted Spearman correlation",
        fun=_masked_spearman,
        reference="Ghazanfar, S., Lin, Y., Su, X., Lin, D.M., Patrick, E., Han, Z.G., Marioni, J.C. and Yang, J.Y.H., 2020."
        "Investigating higher-order interactions in single-cell data with scHOT. Nature methods, 17(8), pp.799-806.",
    ),
]
