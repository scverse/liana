import warnings
from collections.abc import Callable

import numpy as np
import pytest
from anndata import AnnData
from fast_array_utils.conv import to_dense
from tests._helpers import as_frame, get_csr, get_layer_csr, get_obs, get_raw_csr, get_x

from liana._core._pipe_utils._pre import assert_covered, prep_check_adata
from liana.method import cellphonedb, natmi


def test_prep_check_adata(pbmc68k: AnnData) -> None:
    temp = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=0, use_raw=True, layer=None)
    np.testing.assert_almost_equal(np.sum(get_csr(temp).data), 319044.22, decimal=1)

    # By name and dense: a CSR is free to store its columns in any order within a row, so `.data` alone says nothing about which gene a value belongs to.
    desired = np.array([[0.0, 1.591, 2.544, 0.0, 0.0, 2.177], [0.0, 0.0, 2.496, 0.0, 0.0, 0.0]])
    corner = temp[temp.obs_names[:2], temp.var_names[:6]]
    np.testing.assert_almost_equal(to_dense(get_x(corner)), desired, decimal=3)

    # test filtering
    filt = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=20, use_raw=True)
    assert len(filt.obs["@label"]) == 660


def test_default_reads_X_not_raw(pbmc68k: AnnData) -> None:
    # Guards the public default: `use_raw` defaults to False, so methods read `.X`.
    # pbmc68k_reduced ships scaled (negative) data in `.X`, which the single-cell pipe now
    # rejects, so put a doubled copy of the log-normalised `.raw` there instead: non-negative
    # and log1p-plausible, but still not equal to `.raw` -- which is what the test needs.
    pbmc68k.X = get_raw_csr(pbmc68k) * 2
    assert get_x(pbmc68k).min() >= 0

    default = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False))
    from_x = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False, use_raw=False))
    from_raw = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False, use_raw=True))

    assert default.equals(from_x)  # default == .X
    assert not default.equals(from_raw)  # and differs from .raw


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(
            lambda adata: prep_check_adata(adata=adata, groupby="bulk_labels", min_cells=5, block_negatives=True),
            id="prep_check_adata",
        ),
        # the pipe is the funnel: every `li.mt` single-cell method inherits the check
        pytest.param(
            lambda adata: cellphonedb(adata, groupby="bulk_labels", n_perms=None, inplace=False),
            id="cellphonedb",
        ),
    ],
)
def test_block_negatives_raises(pbmc68k: AnnData, run: Callable[[AnnData], object]) -> None:
    # pbmc68k_reduced ships scaled (centred) data in .X. `sqrt`, `log2` and `gmean` of a
    # negative mean are all NaN, so the single-cell pipe refuses it rather than scoring it.
    assert get_x(pbmc68k).min() < 0

    with pytest.raises(ValueError, match="negative values"):
        run(pbmc68k)


def test_block_negatives_off_by_default(pbmc68k: AnnData) -> None:
    # off by default, so the signed-data callers (bivariate, misty, LRIC) are unaffected
    prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5)
    # and log-normalised input passes with the check on
    prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, use_raw=True, block_negatives=True)


def test_unlabelled_cells_raise(pbmc68k: AnnData) -> None:
    """An unlabelled cell used to be folded into the first label's permuted mean."""
    labels = get_obs(pbmc68k)["bulk_labels"].copy()
    labels.iloc[:50] = np.nan
    get_obs(pbmc68k)["bulk_labels"] = labels

    with pytest.raises(ValueError, match="50 of 700 cells carry no `bulk_labels` label"):
        prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5)

    # the guard sits in the pipe's preprocessing, so the non-permutation methods raise it too --
    # they used to drop the unlabelled cells silently
    with pytest.raises(ValueError, match="carry no `bulk_labels` label"):
        natmi(pbmc68k, groupby="bulk_labels", n_perms=None, use_raw=True, inplace=False)


def _counts(seed: int = 0) -> np.ndarray:
    """A sparse integer count matrix, the shape of input the heuristic has to recognise."""
    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(3, 0.3, (200, 100)).astype("float32")
    X[rng.random(X.shape) < 0.6] = 0
    return X


def _cp10k(X: np.ndarray) -> np.ndarray:
    return np.asarray(X / np.maximum(X.sum(1, keepdims=True), 1e-9) * 1e4)


def _lognorm_warnings(X: np.ndarray, *, block_negatives: bool, check_lognorm: bool = True) -> list[str]:
    adata = AnnData(X)
    adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prep_check_adata(
            adata=adata,
            groupby=None,
            min_cells=None,
            block_negatives=block_negatives,
            check_lognorm=check_lognorm,
        )
    return [str(w.message) for w in caught if "log1p-normalised" in str(w.message)]


@pytest.mark.parametrize(
    ("name", "build", "expect"),
    [
        ("log-normalised", lambda X: np.log1p(_cp10k(X)), None),
        ("raw counts", lambda X: X, "its values are all integers"),
        ("shallow counts", lambda X: np.minimum(X, 3.0), "its values are all integers"),
        ("normalised, not logged", _cp10k, "its maximum is"),
        # a binary indicator and a constant matrix say nothing about normalisation -- the spatial
        # methods pass both legitimately, so neither may trip the integer test
        ("binary indicator", lambda X: (X > 0).astype("float32"), None),
        ("constant", lambda X: np.ones_like(X), None),
        ("empty", lambda X: np.zeros_like(X), None),
    ],
)
@pytest.mark.parametrize("block_negatives", [True, False])
def test_lognorm_heuristic(
    name: str, build: Callable[[np.ndarray], np.ndarray], expect: str | None, block_negatives: bool
) -> None:
    hits = _lognorm_warnings(build(_counts()), block_negatives=block_negatives)
    # the spatial and bivariate methods take signed data of any scale, so the advice never applies
    # to them -- whatever the matrix looks like
    if expect is None or not block_negatives:
        assert not hits, f"{name} must not warn, got {hits}"
    else:
        assert len(hits) == 1 and expect in hits[0], f"{name}: {hits}"


def test_check_lognorm_off(pbmc68k: AnnData) -> None:
    # `lric` takes the rejection without the advice: its weights are a ratio to a per-gene
    # mean, so negatives are fatal but any positive rescaling cancels
    assert not _lognorm_warnings(_counts(), block_negatives=True, check_lognorm=False)
    assert get_x(pbmc68k).min() < 0
    with pytest.raises(ValueError, match="negative values"):
        prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, block_negatives=True, check_lognorm=False)


def test_check_if_covered(pbmc68k: AnnData) -> None:
    with pytest.raises(ValueError):
        assert_covered(["NOT", "HERE"], pbmc68k.var_names, verbose=True)


def test_choose_mtx(pbmc68k: AnnData) -> None:
    # check if default is used correctly
    raw_adata = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5)
    assert np.min(get_csr(raw_adata).data) < 0

    # check if correct layer is returned
    by_layer = prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, use_raw=True)
    by_layer.layers["scaled"] = get_x(by_layer)
    extracted = prep_check_adata(by_layer, groupby="bulk_labels", min_cells=5, layer="scaled")

    np.testing.assert_almost_equal(get_layer_csr(by_layer, "scaled").data, get_csr(extracted).data)


def test_choose_mtx_failure(pbmc68k: AnnData) -> None:
    pbmc68k.layers["scaled_counts"] = get_x(pbmc68k)
    # check exception if both layer and use_raw are provided
    with pytest.raises(ValueError):
        prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, layer="scaled_counts", use_raw=True)

    # check exception if .raw is not initialized
    del pbmc68k.raw
    with pytest.raises(ValueError):
        prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, use_raw=True)
