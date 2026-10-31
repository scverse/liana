import pathlib

import numpy as np
import pytest
from anndata import AnnData
from pandas import DataFrame, read_csv
from tests._helpers import get_obs, get_raw_x, get_x

from liana._core._pipe_utils import _get_mean_perms
from liana._core._pipe_utils._get_mean_perms import (
    _calculate_pvals,
    _get_mat_idx,
    _get_means_perms,
    _PermStatChunks,
)
from liana.method.sc._cellphonedb import _mean


@pytest.fixture
def adata(pbmc68k: AnnData) -> AnnData:
    """`pbmc68k_reduced` as the pipe expects it: `.raw` (log-normalised)
    expression in `.X`, labels in `@label`.
    """
    pbmc68k.X = get_raw_x(pbmc68k)
    get_obs(pbmc68k)["@label"] = get_obs(pbmc68k)["bulk_labels"]

    return pbmc68k


@pytest.fixture
def all_defaults(data_dir: pathlib.Path) -> DataFrame:
    """The liana_pipe output obtained with all default parameters."""
    return read_csv(data_dir / "all_defaults.csv", index_col=0)


def test_perms(adata: AnnData) -> None:
    perms = _get_means_perms(
        adata=adata, norm_factor=None, aggregation="mean", n_perms=100, seed=1337, n_jobs=1, verbose=False
    )

    assert perms.shape == (100, 10, 765)


def test_positions(adata: AnnData, all_defaults: DataFrame) -> None:
    ligand_idx, receptor_idx, source_idx, target_idx = _get_mat_idx(adata, all_defaults)

    labels = get_obs(adata)["@label"].cat.categories

    assert set(ligand_idx[all_defaults["ligand"] == "MIF"]) == {740}
    assert set(receptor_idx[all_defaults["receptor"] == "CD4"]) == {465}
    assert set(source_idx[all_defaults["source"] == "Dendritic"]) == {labels.get_loc("Dendritic")}
    assert set(target_idx[all_defaults["target"] == "Dendritic"]) == {labels.get_loc("Dendritic")}


def test_positions_absent_gene(adata: AnnData, all_defaults: DataFrame) -> None:
    unknown = all_defaults.copy()
    unknown.loc[unknown.index[0], "ligand"] = "NOT_A_GENE"

    with pytest.raises(KeyError, match="ligand\\(s\\) absent from `adata.var_names`: NOT_A_GENE"):
        _get_mat_idx(adata, unknown)


def test_perms_are_parallel_invariant(adata: AnnData) -> None:
    serial = _get_means_perms(
        adata=adata, norm_factor=None, aggregation="mean", n_perms=20, seed=1337, n_jobs=1, verbose=False
    )
    parallel = _get_means_perms(
        adata=adata, norm_factor=None, aggregation="mean", n_perms=20, seed=1337, n_jobs=2, verbose=False
    )

    np.testing.assert_array_equal(serial, parallel)


def test_cellchat_perms(adata: AnnData) -> None:
    mat_max = np.float32(get_x(adata).max())  # float32, as the pipeline computes it

    perms = _get_means_perms(
        adata=adata, norm_factor=None, aggregation="trimean", n_perms=100, seed=1337, n_jobs=1, verbose=False
    )

    assert perms.shape == (100, 10, 765)

    desired = np.array(
        [33840.83, 36332.442, 34569.577, 33819.275, 33809.956, 33785.234, 33844.524, 34986.043, 34304.404, 33644.323]
    )
    expected = perms.sum(axis=0).sum(axis=1)

    np.testing.assert_almost_equal(expected, desired, decimal=3)

    perms = _get_means_perms(
        adata=adata, norm_factor=mat_max, aggregation="trimean", n_perms=100, seed=1337, n_jobs=1, verbose=False
    )
    desired = np.array(
        [
            5215.107487,
            5599.082231,
            5327.412358,
            5211.785598,
            5210.349528,
            5206.53966,
            5215.676758,
            5391.592763,
            5286.547464,
            5184.824284,
        ]
    )
    expected = perms.sum(axis=0).sum(axis=1)

    np.testing.assert_almost_equal(expected, desired, decimal=6)


def test_perm_stat_chunks_match_whole_tensor(
    adata: AnnData, all_defaults: DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocks concatenate to the whole tensor, and p-values do not depend on the block size."""
    perms = _get_means_perms(
        adata=adata, norm_factor=None, aggregation="mean", n_perms=23, seed=1337, n_jobs=1, verbose=False
    )
    ligand_idx, receptor_idx, source_idx, target_idx = _get_mat_idx(adata, all_defaults)
    whole = np.stack((perms[:, source_idx, ligand_idx], perms[:, target_idx, receptor_idx]), axis=0)
    truth = _mean(whole[:, 0, :])  # any per-interaction reference works; ties are exercised too
    chunks = _PermStatChunks(perms, source_idx, ligand_idx, target_idx, receptor_idx)

    # reference: the whole-tensor computation the chunked loop replaced
    stat = _mean(whole, axis=0)
    expected = (np.greater_equal(stat, truth) | np.isclose(stat, truth, rtol=1e-6, atol=0.0)).sum(axis=0) / 23

    single = _calculate_pvals(truth, chunks, _mean)
    assert len(list(chunks)) == 1
    np.testing.assert_array_equal(single, expected)

    # cap so that 5 permutations fit a block: 23 perms -> blocks of 5, 5, 5, 5, 3
    monkeypatch.setattr(_get_mean_perms, "_MAX_PERM_STAT_ELEMENTS", 5 * ligand_idx.size)
    blocks = list(chunks)
    assert [b.shape[1] for b in blocks] == [5, 5, 5, 5, 3]
    np.testing.assert_array_equal(np.concatenate(blocks, axis=1), whole)
    np.testing.assert_array_equal(_calculate_pvals(truth, chunks, _mean), single)
