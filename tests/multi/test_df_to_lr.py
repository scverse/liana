import pytest
from anndata import AnnData
from pandas import DataFrame

from liana.datasets._sample_dea import _sample_dea
from liana.method import df_to_lr

groupby = "bulk_labels"


@pytest.fixture
def dea_df(toy_adata: AnnData) -> DataFrame:
    """Toy differential expression results, as e.g. decoupler would return them."""
    return _sample_dea(toy_adata, groupby)


def test_dea_to_lr(toy_adata: AnnData, dea_df: DataFrame) -> None:
    lr_res = df_to_lr(
        toy_adata,
        dea_df=dea_df,
        resource_name="consensus",
        expr_prop=0.1,
        groupby="bulk_labels",
        stat_keys=["stat", "pval", "padjusted"],
        use_raw=False,
        complex_col=None,
        verbose=True,
        min_cells=10,
        return_all_lrs=False,
    )
    assert lr_res.shape == (139, 22)
    # assert ligand_stat, ligand_pval, and ligand_padjusted are in lr_res.columns
    columns = lr_res.columns
    expected_columns = [
        "ligand",
        "ligand_stat",
        "ligand_pval",
        "ligand_padjusted",
        "ligand_expr",
        "receptor",
        "receptor_stat",
        "receptor_pval",
        "receptor_padjusted",
        "receptor_expr",
    ]
    for col in expected_columns:
        assert col in columns
    assert lr_res["interaction_padjusted"].mean() == 0.56700673783093


def test_dea_to_lr_params(toy_adata: AnnData, dea_df: DataFrame) -> None:
    lr_res = df_to_lr(
        toy_adata,
        dea_df=dea_df,
        expr_prop=0.1,
        min_cells=10,
        groupby="bulk_labels",
        stat_keys=["stat", "pval", "padjusted"],
        use_raw=False,
        complex_col="stat",
        verbose=True,
        return_all_lrs=True,
    )
    assert lr_res.shape == (3321, 23)


def test_dea_to_lr_shared_stat_suffix(toy_adata: AnnData, dea_df: DataFrame) -> None:
    """`stat_keys` sharing a suffix must not be averaged into each other's `interaction_` column.

    A suffix match made `"val"` pick up `ligand_pval`/`receptor_pval` and the `interaction_pval`
    written one iteration earlier, so `interaction_val` was neither of the two `*_val` columns.
    """
    dea_df = dea_df.rename(columns={"stat": "val"})

    lr_res = df_to_lr(
        toy_adata,
        dea_df=dea_df,
        expr_prop=0.1,
        min_cells=10,
        groupby=groupby,
        stat_keys=["pval", "val"],
        use_raw=False,
        verbose=False,
    )

    expected = lr_res[["ligand_val", "receptor_val"]].mean(axis=1)
    assert lr_res["interaction_val"].equals(expected)
    assert lr_res["interaction_pval"].equals(lr_res[["ligand_pval", "receptor_pval"]].mean(axis=1))
