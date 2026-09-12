"""Functions to deal with protein complexes"""

from __future__ import annotations

import pandas as pd

from liana._core._docs import d


@d.dedent
def _filter_reassemble_complexes(
    lr_res: pd.DataFrame,
    _key_cols: list[str],
    complex_cols: list[str],
    expr_prop: float,
    return_all_lrs: bool = False,
    complex_policy: str = "min",
) -> pd.DataFrame:
    """
    Reassemble complexes from exploded long-format pandas Dataframe.

    Parameters
    ----------
    lr_res
        long-format pandas dataframe with exploded complex subunits
    _key_cols
        primary key for lr_res, typically a list with the following elements -
        ['source', 'target', 'ligand_complex', 'receptor_complex']
    complex_cols
        method/complex-relevant columns
    %(expr_prop)s
    %(return_all_lrs)s
    complex_policy
        approach by which the complexes are reassembled

    Raises
    ------
    ValueError
        If no ligand-receptor pair passes `expr_prop`.

    Return
    -----------
    lr_res: a reduced long-format pandas dataframe
    """
    # Filter by expr_prop (inner join only complexes where all subunits are expressed)
    expressed = (
        lr_res[_key_cols + ["ligand_props", "receptor_props"]]
        .set_index(_key_cols)
        .stack()
        .groupby(_key_cols)
        .agg(prop_min=complex_policy)
        .reset_index()
    )
    kept = expressed[expressed["prop_min"] >= expr_prop]
    # an empty `expressed` leaves the callers to fail obscurely much later (a `KeyError` on a
    # score column, an `IndexError` while indexing the permutation tensor, or silently 0 rows),
    # and `return_all_lrs=True` hides it further by re-appending every filtered-out interaction
    if kept.empty and not expressed.empty:
        raise ValueError(
            f"No ligand-receptor pair passed `expr_prop={expr_prop}`: the highest minimum subunit "
            f"proportion is {expressed['prop_min'].max():.3g}. Lower `expr_prop`. Note that "
            "`return_all_lrs=True` does not help here -- there is no score to fill the rest with."
        )
    expressed = kept

    if not return_all_lrs:
        lr_res = lr_res.merge(expressed, how="inner", on=_key_cols)
    else:
        expressed["lrs_to_keep"] = True
        lr_res = lr_res.merge(expressed, how="left", on=_key_cols)
        # chained inplace assignment is a no-op under copy-on-write
        # `~` on an object-dtype bool is a bitwise not, so `~True == -2`
        lr_res["lrs_to_keep"] = lr_res["lrs_to_keep"].fillna(value=False).astype(bool)
        lr_res["prop_min"] = lr_res["prop_min"].fillna(value=0)

    # check if complex policy is only min
    aggs = {complex_policy, "min"}

    for col in complex_cols:
        lr_res = _reduce_complexes(col=col, lr_res=lr_res, key_cols=_key_cols, aggs=aggs)

    # Each `_reduce_complexes` pass keeps only the rows tied at that column's per-key minimum, so a
    # key survives more than once only with identical `complex_cols` values -- ties, from which any
    # row will do. Downstream expects one row per key.
    lr_res = lr_res.drop_duplicates(subset=_key_cols, keep="first")

    return lr_res


def _reduce_complexes(
    col: str,
    lr_res: pd.DataFrame,
    key_cols: list[str],
    aggs: set[str],
) -> pd.DataFrame:
    grouped = lr_res.groupby(key_cols)

    # Get min cols by which we will join
    # then rename from agg name to column name (e.g. 'min' to 'ligand_min')
    lr_min = (
        grouped[col]
        .agg(list(aggs))
        .reset_index()
        .copy()
        .rename(columns={agg: col.split("_")[0] + "_" + agg for agg in aggs})
    )

    # right is the min subunit for that column
    join_key = col.split("_")[0] + "_min"  # ligand_min or receptor_min

    # Here, I join the min value and keep only those rows that match
    merged = lr_res.merge(lr_min, on=key_cols, how="inner")

    return merged[merged[col] == merged[join_key]].drop(join_key, axis=1)


def _explode_complexes(resource: pd.DataFrame, SOURCE: str = "ligand", TARGET: str = "receptor") -> pd.DataFrame:
    """
    Function to explode ligand-receptor complexes

    Parameters
    ----------
    resource
        Ligand-receptor resource
    SOURCE
        Name of the source (typically 'ligand') column
    TARGET
        Name of the target (typically 'receptor') column

    Returns
    -------
    A resource with exploded complexes
    """
    resource["interaction"] = resource[SOURCE] + "&" + resource[TARGET]
    resource = (
        resource.set_index("interaction")
        .apply(lambda x: x.str.split("_"))
        .explode([TARGET])
        .explode(SOURCE)
        .reset_index()
    )
    resource[[f"{SOURCE}_complex", f"{TARGET}_complex"]] = resource["interaction"].str.split("&", expand=True)

    return resource
