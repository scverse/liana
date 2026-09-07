from itertools import product

import numpy as np
import pandas as pd
from anndata import AnnData

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._constants import InternalValues as I
from liana._core._constants import PrimaryColumns as P
from liana._core._docs import d
from liana._core._pipe_utils import assert_covered, filter_resource, prep_check_adata
from liana._core._pipe_utils._common import _get_groupby_subset, _get_props, _join_stats
from liana._core._pipe_utils._pre import _check_groupby, _choose_mtx_rep
from liana._core._types import get_obs
from liana.resource._reassemble_complexes import _explode_complexes, _filter_reassemble_complexes
from liana.resource.select_resource import _handle_resource


@d.dedent
def df_to_lr(
    adata: AnnData,
    dea_df: pd.DataFrame,
    groupby: str,
    stat_keys: list[str],
    resource_name: str = V.resource_name,
    resource: pd.DataFrame | None = V.resource,
    interactions: list[tuple[str, str]] | None = V.interactions,
    groupby_pairs: pd.DataFrame | None = V.groupby_pairs,
    layer: str | None = V.layer,
    use_raw: bool = V.use_raw,
    expr_prop: float = V.expr_prop,
    min_cells: int = V.min_cells,
    complex_col: str | None = None,
    return_all_lrs: bool = V.return_all_lrs,
    source_labels: list[str] | None = None,
    target_labels: list[str] | None = None,
    lr_sep: str = V.lr_sep,
    verbose: bool = V.verbose,
) -> pd.DataFrame:
    """
    Convert DEA results to ligand-receptor pairs.

    Parameters
    ----------
    %(adata)s
    dea_df
        DEA results. Index must match adata.var_names
    %(groupby)s
    stat_keys
        List of statistics to be used for ligand-receptor pairs
    %(resource_name)s
    %(resource)s
    %(interactions)s
    %(groupby_pairs)s
    %(layer)s
    %(use_raw)s
    %(expr_prop)s
    %(min_cells)s
    complex_col
        Column in `dea_df` to use for complex expression. Default is None.
        If None, will use mean expression ('expr') calculated per group in `groupby`.
    %(return_all_lrs)s
    %(source_labels)s
    %(target_labels)s
    %(lr_sep)s
    %(verbose)s

    Returns
    -------
    Returns a pd.DataFrame with joined ligand-receptor pairs and statistics.

    Raises
    ------
    ValueError
        If the `groupby` value is not in `adata` or `dea_df`, if `dea_df` indexes do not match `adata.var_names` or if `complex_col` does not match one of the computed stats.
    AssertionError
        If there's no match when grouping-by between `adata.obs` and `dea_df`.

    Examples
    --------
    `dea_df` holds per-cell-type differential expression statistics, indexed by
    gene. It normally comes from a tool such as `pydeseq2` or `scanpy`; a stand-in
    is built here so the example stays offline:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> groups = adata.obs["bulk_labels"].cat.categories
    >>> rng = np.random.default_rng(1337)
    >>> dea_df = pd.DataFrame(
    ...     {"bulk_labels": np.repeat(groups, adata.n_vars), "stat": rng.normal(size=len(groups) * adata.n_vars)},
    ...     index=np.tile(adata.var_names, len(groups)),
    ... )
    >>> lr_res = li.mt.df_to_lr(adata, dea_df=dea_df, groupby="bulk_labels", stat_keys=["stat"])

    Each statistic named in `stat_keys` is carried over to both sides of every
    interaction -- as `ligand_stat` and `receptor_stat` -- and averaged into an
    `interaction_stat` column.
    """
    _check_groupby(adata=adata, groupby=groupby, verbose=verbose)
    if (groupby not in adata.obs.columns) or (groupby not in dea_df.columns):
        raise ValueError("groupby must match a column in both adata.obs and dea_df")
    if not np.any(adata.var_names.isin(dea_df.index)):
        raise ValueError("index of dea_df must match adata.var_names")
    if len(np.intersect1d(adata.obs[groupby].unique(), dea_df[groupby].unique())) == 0:
        raise ValueError("`groupby` intersect between `dea_df` and `adata` is 0. Please check `groupby`.")

    resource = _handle_resource(
        interactions=interactions, resource=resource, resource_name=resource_name, verbose=verbose
    )

    stat_names = ["expr", "props"] + stat_keys
    if complex_col is not None:
        if complex_col not in stat_names:
            raise ValueError(
                f"complex_col must be one of `stat_keys`:{stat_keys} or the stats calculated by default: {stat_names}!"
            )
        stat_names = stat_names[stat_names.index(complex_col) :] + stat_names[: stat_names.index(complex_col)]
    else:
        complex_col = "expr"

    groupby_subset = _get_groupby_subset(groupby_pairs=groupby_pairs)

    # Check and Reformat Mat if needed
    adata = prep_check_adata(
        adata=adata,
        groupby=groupby,
        groupby_subset=groupby_subset,
        min_cells=min_cells,
        use_raw=use_raw,
        layer=layer,
        verbose=verbose,
    )

    # reduce dim of adata
    intersect = np.intersect1d(adata.var_names, dea_df.index)
    if intersect.shape[0] != adata.shape[1]:
        _logg("Features in adata and dea_df are mismatched.", verbose=verbose, level="warn")
    adata = adata[:, intersect]

    obs = get_obs(adata)
    labels = obs[I.label].cat.categories
    dedict: dict[str, pd.DataFrame] = {}
    for label in labels:
        temp = adata[obs[I.label] == label, :]
        # `prep_check_adata` above guarantees a csr matrix.
        temp_x = _choose_mtx_rep(temp)
        props = _get_props(temp_x)
        means = np.array(temp_x.mean(axis=0), dtype="float32").flatten()

        stats = (
            pd.DataFrame({"names": temp.var_names, "props": props, "expr": means})
            .assign(label=label)
            .sort_values("names")
        )

        # merge DEA results to props & means
        dea_df.index.name = None
        df = dea_df[dea_df[groupby] == label].drop(groupby, axis=1).reset_index().rename(columns={"index": "names"})

        if not return_all_lrs:
            stats = stats.merge(df, on="names")
        else:
            stats = df.merge(stats, on="names", how="outer")

        dedict[label] = stats[["names", "label", *stat_names]]
        all_stats = pd.concat(dedict.values())

    # Create df /w cell identity pairs
    pairs = pd.DataFrame(list(product(labels, labels))).rename(columns={0: "source", 1: "target"})

    if groupby_pairs is not None:
        pairs = pairs.merge(groupby_pairs, on=[P.source, P.target], how="inner")

    if source_labels is not None:
        pairs = pairs[pairs["source"].isin(source_labels)]
    if target_labels is not None:
        pairs = pairs[pairs["target"].isin(target_labels)]

    resource = _explode_complexes(resource)

    # Check overlap between resource and adata
    assert_covered(
        np.union1d(np.unique(resource["ligand"]), np.unique(resource["receptor"])), all_stats["names"], verbose=verbose
    )

    # Filter Resource
    resource = filter_resource(resource, all_stats["names"].unique())

    # Join Stats to LR
    lr_res = pd.concat(
        [
            _join_stats(source, target, dedict, resource)
            for source, target in zip(pairs["source"], pairs["target"], strict=False)
        ],
    )

    # ligand_ or receptor + stat_keys
    complex_cols = [f"{prefix}_{complex_col}" for prefix in ("ligand", "receptor")]

    # assign receptor and ligand absolutes, NOTE deals with missing values
    _placeholders = ["ligand_absolute", "receptor_absolute"]
    lr_res[_placeholders] = lr_res[complex_cols].apply(lambda x: x.abs())
    if return_all_lrs:
        lr_res[_placeholders] = lr_res[_placeholders].fillna(0)

    lr_res = _filter_reassemble_complexes(
        lr_res=lr_res,
        _key_cols=P.primary,
        expr_prop=expr_prop,
        return_all_lrs=return_all_lrs,
        complex_cols=_placeholders,
    )
    lr_res = lr_res.drop(["prop_min", "interaction", *_placeholders], axis=1)

    # summarise stats for each lr
    for key in stat_names:
        stat_columns = lr_res.columns[lr_res.columns.str.endswith(key)]
        lr_res.loc[:, f"interaction_{key}"] = lr_res.loc[:, stat_columns].mean(axis=1)

    lr_res["interaction"] = lr_res["ligand_complex"] + lr_sep + lr_res["receptor_complex"]

    return lr_res
