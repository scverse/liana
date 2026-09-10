import numpy as np
import pytest
from anndata import AnnData
from fast_array_utils.conv import to_dense
from tests._helpers import as_frame, get_csr, get_layer_csr, get_raw_csr, get_x

from liana._core._pipe_utils._pre import assert_covered, prep_check_adata


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
    from liana.method import cellphonedb

    pbmc68k.X = get_raw_csr(pbmc68k).expm1()
    assert get_x(pbmc68k).min() >= 0

    default = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False))
    from_x = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False, use_raw=False))
    from_raw = as_frame(cellphonedb(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False, use_raw=True))

    assert default.equals(from_x)  # default == .X
    assert not default.equals(from_raw)  # and differs from .raw


def test_block_negatives_raises(pbmc68k: AnnData) -> None:
    # pbmc68k_reduced ships scaled (centred) data in .X. `sqrt`, `log2` and `gmean` of a
    # negative mean are all NaN, so the single-cell pipe refuses it rather than scoring it.
    assert get_x(pbmc68k).min() < 0

    with pytest.raises(ValueError, match="negative values"):
        prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, block_negatives=True)

    # off by default, so the signed-data callers (bivariate, misty, LRIC) are unaffected
    prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5)
    # and log-normalised input passes with the check on
    prep_check_adata(adata=pbmc68k, groupby="bulk_labels", min_cells=5, use_raw=True, block_negatives=True)


def test_sc_methods_reject_negatives(pbmc68k: AnnData) -> None:
    # the pipe is the funnel: every `li.mt` single-cell method inherits the check
    from liana.method import cellphonedb, rank_aggregate

    for method in (cellphonedb, rank_aggregate):
        with pytest.raises(ValueError, match="negative values"):
            method(pbmc68k, groupby="bulk_labels", n_perms=None, inplace=False)


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
