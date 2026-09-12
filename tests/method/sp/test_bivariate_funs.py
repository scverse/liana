from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import rankdata

from liana.method.sp._bivariate._global_functions import GlobalFunction, _global_r
from liana.method.sp._bivariate._local_functions import (
    LocalFunction,
    LocalStat,
    _local_morans,
    _masked_spearman,
    _midranks,
    _norm_product,
    _product,
    _vectorized_cosine,
    _vectorized_jaccard,
    _vectorized_pearson,
    _vectorized_spearman,
)


@pytest.fixture
def mats() -> SimpleNamespace:
    """Two 20x5 feature matrices and a 20x20 weight matrix."""
    rng = np.random.default_rng(seed=0)

    return SimpleNamespace(
        x_mat=rng.normal(size=(20, 5)).astype(np.float32),
        y_mat=rng.normal(size=(20, 5)).astype(np.float32),
        weight=csr_matrix(rng.uniform(size=(20, 20)).astype(np.float32)),
    )


@pytest.fixture
def pval_mats() -> SimpleNamespace:
    """Two 10x10 feature matrices and a weight matrix scaled to sum to `n`."""
    seed = 0
    rng = np.random.default_rng(seed=seed)

    dist = csr_matrix(rng.normal(size=(10, 10)))
    norm_factor = dist.shape[0] / dist.sum()

    return SimpleNamespace(
        seed=seed,
        rng=rng,
        weight=csr_matrix(norm_factor * dist),
        x_mat=rng.normal(size=(10, 10)),
        y_mat=rng.normal(size=(10, 10)),
        n_perms=100,
        mask_negatives=True,
    )


def _assert_bivariate(
    function: LocalStat,
    desired: np.ndarray,
    mats: SimpleNamespace,
    dense_weight: bool = False,
) -> None:
    weight = mats.weight.toarray() if dense_weight else mats.weight
    actual = function(mats.x_mat, mats.y_mat, weight)
    assert actual.shape == (20, 5)
    np.testing.assert_almost_equal(actual[0, :], desired, decimal=5)


# ── local functions ───────────────────────────────────────────────────────────


def test_pc_vectorized(mats: SimpleNamespace) -> None:
    pc_vec_truth = np.array([0.25005114, 0.04262733, -0.00130362, 0.2903336, -0.1236529])
    _assert_bivariate(_vectorized_pearson, pc_vec_truth, mats)


def test_sp_vectorized(mats: SimpleNamespace) -> None:
    sp_vec_truth = np.array([0.23636213, 0.16480759, -0.01487235, 0.22840601, -0.11492937])
    _assert_bivariate(_vectorized_spearman, sp_vec_truth, mats)


def test_sp_masked(mats: SimpleNamespace) -> None:
    sp_masked_truth = np.array([0.23636216, 0.16480756, -0.0148723, 0.22840606, -0.11492944])
    _assert_bivariate(_masked_spearman, sp_masked_truth, mats, dense_weight=True)


@pytest.mark.parametrize("tied", [False, True])
def test_midranks_matches_scipy(tied: bool) -> None:
    """``_masked_spearman``'s ranking must average ties, as `scipy.stats.rankdata` does."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=100).astype(np.float32)
    if tied:
        x = (x * (rng.random(100) > 0.5)).round(1).astype(np.float32)
        assert len(np.unique(x)) < x.size, "fixture must carry ties"

    np.testing.assert_allclose(_midranks(x), rankdata(x, method="average"), rtol=1e-6)


def test_sp_masked_is_invariant_to_cell_order() -> None:
    """Relabelling the cells must permute the scores and nothing else.

    `_wcorr` used to take *ordinal* ranks (``argsort().argsort()``), which split tied
    values in whatever order the sort visited them -- so on zero-inflated data, i.e. any
    expression matrix, the statistic moved when the cells were reordered. A revert fails
    this by ~0.6 on the fixture below.
    """
    rng = np.random.default_rng(1)
    n, xy_n = 40, 3
    # zero-inflated, as expression data is -- without ties the defect is invisible
    x_mat = (rng.normal(size=(n, xy_n)) * (rng.random((n, xy_n)) > 0.5)).astype(np.float32)
    y_mat = (rng.normal(size=(n, xy_n)) * (rng.random((n, xy_n)) > 0.5)).astype(np.float32)
    weight = (rng.random((n, n)) * (rng.random((n, n)) < 0.3)).astype(np.float32)

    assert (x_mat == 0).sum() > n, "fixture must carry ties"

    base = _masked_spearman(x_mat, y_mat, weight)
    assert (np.abs(base) > 0.1).sum() > 5, "fixture must carry signal, else this is vacuous"

    perm = rng.permutation(n)
    actual = _masked_spearman(x_mat[perm], y_mat[perm], np.ascontiguousarray(weight[np.ix_(perm, perm)]))

    np.testing.assert_allclose(actual, base[perm], atol=1e-6)


def test_sp_masked_scores_a_constant_neighbourhood_zero() -> None:
    """A feature that is constant within a neighbourhood must score 0 there, never nan.

    Midranks make an all-tied neighbourhood *exactly* constant, so its weighted variance is
    mathematically 0 -- but in float32 with non-integer (gaussian-like) weights it rounds to
    a tiny value of either sign: negative made ``(dx*dy)**0.5`` nan, positive gave 0/~0
    garbage. A nan then reads as "not >=" in `_permutation_pvals`, handing an unexpressed
    gene p = 0. Without the variance guard this fixture yields 135 nans.
    """
    rng = np.random.default_rng(0)
    n, xy_n = 60, 4
    # a 1D gaussian kernel: float, non-binary weights over a band of ~6-11 neighbours
    d2 = (np.arange(n)[:, None] - np.arange(n)[None, :]) ** 2
    weight = np.exp(-d2 / (2 * 4.0**2)).astype(np.float32)
    weight[d2 > 5**2] = 0.0
    weight = np.ascontiguousarray(weight)

    x_mat = (rng.normal(size=(n, xy_n)) * (rng.random((n, xy_n)) > 0.4)).astype(np.float32)
    # y is zero -- hence constant -- inside every neighbourhood below cell ~35, but not globally
    y_mat = np.zeros((n, xy_n), dtype=np.float32)
    y_mat[40:, :] = rng.normal(size=(n - 40, xy_n)).astype(np.float32)

    nbrs = weight > 0
    constant = np.array(
        [[np.ptp(x_mat[nbrs[i], j]) == 0 or np.ptp(y_mat[nbrs[i], j]) == 0 for j in range(xy_n)] for i in range(n)]
    )
    assert constant.sum() > 10, "fixture must carry constant neighbourhoods"

    actual = _masked_spearman(x_mat, y_mat, weight)

    assert not np.isnan(actual).any()
    np.testing.assert_array_equal(actual[constant], 0.0)
    assert (np.abs(actual[~constant]) > 0.1).sum() > 5, "guard must not flatten the rest"


def test_costine_vectorized(mats: SimpleNamespace) -> None:
    cosine_vec_truth = np.array([0.33806977, 0.03215113, 0.0950243, 0.2957758, -0.10259595])
    _assert_bivariate(_vectorized_cosine, cosine_vec_truth, mats)


def test_vectorized_jaccard(mats: SimpleNamespace) -> None:
    jaccard_vec_truth = np.array([0.34295967, 0.35367563, 0.39685577, 0.41780996, 0.30527356])
    _assert_bivariate(_vectorized_jaccard, jaccard_vec_truth, mats)


# NOTE: spatialdm uses raw counts
def test_morans(mats: SimpleNamespace) -> None:
    sp_morans_truth = np.array([-1.54256, 0.64591, 1.30025, 0.55437, -0.77182])
    _assert_bivariate(_local_morans, sp_morans_truth, mats)


def test_product(mats: SimpleNamespace) -> None:
    product_vec_truth = np.array([5.4518123, -0.7268728, 8.350364, 0.53861964, 1.4466602])
    _assert_bivariate(_product, product_vec_truth, mats, dense_weight=True)


def test_norm_product(mats: SimpleNamespace) -> None:
    product_vec_truth = np.array([0.4081537, -0.03988646, 0.42921585, 0.03255661, 0.08895018])
    _assert_bivariate(_norm_product, product_vec_truth, mats, dense_weight=True)


# ── p-values ──────────────────────────────────────────────────────────────────


def test_local_permutation_pvals(pval_mats: SimpleNamespace) -> None:
    local_morans = LocalFunction._get_instance("morans")
    local_truth = pval_mats.rng.normal(size=(10, 10))

    pvals = local_morans._permutation_pvals(
        x_mat=pval_mats.x_mat,
        y_mat=pval_mats.y_mat,
        local_truth=local_truth,
        weight=pval_mats.weight,
        n_perms=pval_mats.n_perms,
        seed=pval_mats.seed,
        mask_negatives=pval_mats.mask_negatives,
        verbose=False,
    )
    assert pvals.shape == (10, 10)


def test_local_zscore_pvals(pval_mats: SimpleNamespace) -> None:
    local_morans = LocalFunction._get_instance("morans")
    local_truth = pval_mats.rng.normal(size=(10, 10))

    actual = local_morans._zscore_pvals(
        x_mat=pval_mats.x_mat,
        y_mat=pval_mats.y_mat,
        weight=pval_mats.weight,
        local_truth=local_truth,
        mask_negatives=pval_mats.mask_negatives,
        verbose=False,
    )
    assert actual.shape == (10, 10)


def test_global_zscore_pvals(pval_mats: SimpleNamespace) -> None:
    global_stat = pval_mats.rng.normal(size=(10))
    pvals = _global_r._zscore_pvals(
        global_stat=global_stat, weight=pval_mats.weight, mask_negatives=pval_mats.mask_negatives
    )
    assert pvals.shape == (10,)


def test_global_permutation_pvals(pval_mats: SimpleNamespace) -> None:
    global_stat = pval_mats.rng.normal(size=(10))
    pvals = _global_r._permutation_pvals(
        x_mat=pval_mats.x_mat,
        y_mat=pval_mats.y_mat,
        global_stat=global_stat,
        seed=pval_mats.seed,
        n_perms=pval_mats.n_perms,
        mask_negatives=pval_mats.mask_negatives,
        weight=pval_mats.weight,
        verbose=False,
    )
    assert pvals.shape == (10,)


def test_lee_matches_closed_form_on_asymmetric_weight() -> None:
    """Global Lee's L needs ``W.T @ W``; ``W @ W`` silently agrees only for symmetric W.

    Called through ``GlobalFunction.__call__`` on purpose: the operator lives there,
    not in ``_lee_stat``, so pinning the statistic alone leaves the fix untested.
    """
    rng = np.random.default_rng(0)
    n = 40
    # a k-NN style weight: every row keeps 5 random off-diagonal entries -> asymmetric
    dense = np.zeros((n, n))
    for i in range(n):
        nbrs = rng.choice([j for j in range(n) if j != i], size=5, replace=False)
        dense[i, nbrs] = rng.uniform(0.1, 1.0, size=5)
    assert np.abs(dense - dense.T).sum() > 0, "fixture must be asymmetric"

    x = rng.normal(size=(n, 1))
    y = rng.normal(size=(n, 1))
    xy_stats = pd.DataFrame(index=[0])
    GlobalFunction.instances["lee"](
        xy_stats=xy_stats,
        x_mat=x,
        y_mat=y,
        weight=csr_matrix(dense),
        seed=0,
        n_perms=None,
        mask_negatives=False,
        verbose=False,
    )

    # Lee's L (2001, eq. 18): z_y^T W^T W z_x / 1^T W^T W 1, over the same z-scores
    # `__call__` takes -- `_zscore`'s population std
    zx = (x - x.mean(0)) / x.std(0)
    zy = (y - y.mean(0)) / y.std(0)

    def _closed_form(W: np.ndarray) -> float:
        return float(((W @ zx) * zy).sum() / W.sum())

    np.testing.assert_allclose(xy_stats["lee"].to_numpy()[0], _closed_form(dense.T @ dense), rtol=1e-10)
    # what makes the pin above discriminating: on this fixture the two operators
    # disagree, so a revert to `W @ W` fails it (on a symmetric W it would not)
    assert abs(_closed_form(dense @ dense) - _closed_form(dense.T @ dense)) > 1e-6


@pytest.mark.parametrize("set_diag", [False, True])
def test_local_std_matches_a_permutation_null(set_diag: bool) -> None:
    """The analytical null std must match the spread of the statistic under permutation.

    The only shape of test that can catch a wrong variance -- an expression mirroring
    the code cannot. The closed form ``2 (n-1)^2/n^2 s_x^2 s_y^2 (sum_j w_ij^2 + w_ii^2)``
    (``s`` being the ``n/(n-1)``-inflated population std `_zscore_pvals` passes in) is
    compared against the empirical std of ``R_i = x_i(Wy)_i + y_i(Wx)_i`` under
    INDEPENDENT permutations of x and y -- the independence the closed form assumes,
    unlike `_permutation_pvals`, which reuses one index for both.

    Compared as a median over spots and variable pairs: a single spot's empirical std
    carries ~3% Monte-Carlo error, the median far less. The 5% tolerance also absorbs
    the ~2% by which the closed form runs high here -- it treats the permuted values as
    i.i.d. and so drops a finite-population term of order ``neighbours / (n - 1)``.
    A factor-2 error (41%) or a dropped ``w_ii^2`` (16% with the diagonal below) is
    nowhere near that.
    """
    local_morans = LocalFunction._get_instance("morans")
    rng = np.random.default_rng(0)
    n, k, n_pairs, n_perms = 200, 4, 2, 500

    # a k-NN style weight: exactly `k` neighbours each, so no spot has a degenerate null
    weight = np.zeros((n, n))
    for i in range(n):
        weight[i, rng.choice([j for j in range(n) if j != i], size=k, replace=False)] = rng.uniform(0.1, 1.0, size=k)
    # `set_diag=True` in `li.pp.spatial_neighbors` puts the kernel at distance 0 -- i.e. 1
    np.fill_diagonal(weight, 1.0 if set_diag else 0.0)

    x = rng.normal(size=(n, n_pairs)) * np.array([1.0, 3.0])  # unequal scales across pairs
    y = rng.normal(size=(n, n_pairs))
    x, y = x - x.mean(0), y - y.mean(0)  # the statistic's null assumes zero mean

    stats = np.empty((n_perms, n, n_pairs))
    for p in range(n_perms):
        xp, yp = x[rng.permutation(n)], y[rng.permutation(n)]
        stats[p] = xp * (weight @ yp) + yp * (weight @ xp)

    # `_zscore_pvals` fits sigma with `norm.fit` (the ddof=0 MLE) and inflates by n/(n-1)
    sigma = [v.std(axis=0) * n / (n - 1) for v in (x, y)]
    analytical = local_morans._get_local_std(sigma[0], sigma[1], weight, n)

    np.testing.assert_allclose(np.median(stats.std(axis=0) / analytical), 1.0, rtol=0.05)


def test_local_std_sparse_matches_dense() -> None:
    """The sparse path must not change the result -- it only avoids an O(n^2) densification."""
    local_morans = LocalFunction._get_instance("morans")
    rng = np.random.default_rng(2)
    n = 20
    dense = rng.uniform(size=(n, n))
    np.fill_diagonal(dense, 0.3)
    sigma = np.array([1.0, 0.5])

    np.testing.assert_allclose(
        local_morans._get_local_std(sigma, sigma, csr_matrix(dense), n),
        local_morans._get_local_std(sigma, sigma, dense, n),
        rtol=1e-10,
    )


def test_local_analytical_pvals_are_two_sided(pval_mats: SimpleNamespace) -> None:
    """With ``mask_negatives=False`` the p-values must be two-sided, i.e. able to exceed 0.5."""
    local_morans = LocalFunction._get_instance("morans")
    local_truth = np.zeros((10, 10))  # z == 0 everywhere -> a two-sided p-value of exactly 1

    actual = local_morans._zscore_pvals(
        x_mat=pval_mats.x_mat,
        y_mat=pval_mats.y_mat,
        weight=pval_mats.weight,
        local_truth=local_truth,
        mask_negatives=False,
        verbose=False,
    )
    np.testing.assert_allclose(actual, 1.0, rtol=1e-10)


# ── `_norm_max` ─────────────────────────────────────────────────────────────

# col 0: strictly positive -- max-scaling was a no-op here (why the morans pins hold)
# col 1: all-negative -- max-scaling used to flip the sign of every z-score
# col 2: max == 0 -- max-scaling produced inf/nan, silently zeroing the column
_MAT = np.array(
    [
        [1.0, -1.0, 0.0],
        [2.0, -2.0, 0.0],
        [3.0, -3.0, -1.0],
    ]
)


@pytest.mark.parametrize("sparse", [False, True])
def test_norm_max_is_scale_invariant_and_keeps_sign(sparse: bool) -> None:
    """`_norm_max` must not divide by the column max (all-negative flips, max==0 zeroes)."""
    fun = LocalFunction._get_instance("morans")
    X = csr_matrix(_MAT) if sparse else _MAT

    actual = fun._norm_max(X)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == _MAT.shape

    expected = np.array(
        [
            [-1.22474487, 1.22474487, 0.70710678],
            [0.0, 0.0, 0.70710678],
            [1.22474487, -1.22474487, -1.41421356],
        ]
    )
    np.testing.assert_almost_equal(actual, expected, decimal=6)


def test_norm_max_maps_constant_column_to_zero() -> None:
    """A genuinely constant column has no z-score; the nan->0 mapping must survive."""
    fun = LocalFunction._get_instance("morans")

    actual = fun._norm_max(np.array([[5.0], [5.0], [5.0]]))

    np.testing.assert_array_equal(actual, np.zeros((3, 1)))
