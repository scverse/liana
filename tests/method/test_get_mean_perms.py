import pathlib

import numpy as np
import pytest
from anndata import AnnData
from pandas import DataFrame, read_csv
from tests._helpers import get_obs, get_raw_x, get_x

from liana._core._pipe_utils._get_mean_perms import _get_mat_idx, _get_means_perms


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
