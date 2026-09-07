from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData

from liana._core._docs import d
from liana._core._types import get_obs, get_var


@d.dedent
def get_factor_scores(
    adata: AnnData | MuData, obsm_key: str | None = None, obs_keys: list[str] | None = None
) -> pd.DataFrame:
    """
    Extract factor scores from an AnnData object.

    Parameters
    ----------
    %(adata)s
    obsm_key
        Key to use when extracting factor scores from `adata.obsm`
    obs_keys
        List of keys to use when extracting metadata from `adata.obs`
        If None, no metadata is extracted. Default is None.

    Returns
    -------
    Returns a pandas DataFrame with the factor scores.

    Raises
    ------
    ValueError
        If `obsm_key` not in `.obsm`

    Examples
    --------
    `obsm_key` points at the cell-or-sample by factor matrix written by a
    factorization model -- MOFA (`'X_mofa'`), or :func:`liana.ms.nmf` as here:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> lrdata = li.mt.bivariate(adata, resource_name="consensus", local_name="cosine", global_name=None, n_perms=None)
    >>> li.ms.nmf(lrdata, n_components=3, random_state=0)
    >>> scores = li.ms.get_factor_scores(lrdata, obsm_key="NMF_W", obs_keys=["bulk_labels"])

    `scores` has one `Factor{i}` column per factor, an `index` column of the original
    barcodes, and any `.obs` columns named in `obs_keys`.
    """
    if obsm_key is None or obsm_key not in adata.obsm.keys():
        raise ValueError(f"{obsm_key} not found in `.obsm`")

    obs = get_obs(adata)
    df = pd.DataFrame(np.asarray(adata.obsm[obsm_key]), index=obs.index)

    df.columns = [f"Factor{x + 1}" for x in range(df.shape[1])]
    df = df.reset_index()

    # join with metadata
    if obs_keys is not None:
        df = df.merge(obs[obs_keys].reset_index())

    return df


@d.dedent
def get_variable_loadings(
    adata: AnnData | MuData | None = None,
    varm_key: str | None = None,
    view_sep: str | None = None,
    variable_sep: str | None = None,
    pair_sep: str | None = None,
    var_names: list[str] | None = None,
    pair_names: list[str] | None = None,
    drop_columns: bool = True,
    loadings: pd.DataFrame | dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Extract variable loadings from an AnnData object.

    Parameters
    ----------
    %(adata)s
    varm_key
        Key to use when extracting variable loadings from `mdata.varm`.
        Ignored when `loadings` is provided.
    view_sep
        Separator to use when splitting view:variable names into view and variable
    variable_sep
        Separator to use when splitting variable names into `var_names`
    pair_sep
        Separator to use when splitting view names into `pair_names`
    var_names
        Variable names given to the splitted variable ('ligand_complex' and 'receptor_complex' by default)
    pair_names
        Variable names given to the splitted pair ('source' and 'target' by default)
    drop_columns
        If True, drop the `view:variable` column
    loadings
        Pre-extracted loadings to use instead of reading from `adata.varm`.
        Either a features-by-factors :class:`~pandas.DataFrame`, or a dict of
        per-view features-by-factors DataFrames (e.g. the output of a MOFA-Flex
        model's ``get_weights()``), which is concatenated feature-wise. When
        provided, `adata` and `varm_key` are ignored and the existing factor
        column names are preserved.

    Returns
    -------
    Returns a pandas DataFrame with the variable loadings for the specified index.

    Raises
    ------
    ValueError
        If `varm_key` not found in `.varm` (when `loadings` is not provided)

    Examples
    --------
    `varm_key` points at the feature by factor matrix written by a factorization
    model -- here :func:`liana.ms.nmf` on the local scores of
    ``liana.mt.bivariate``, whose `var_names` are `'ligand^receptor'`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> lrdata = li.mt.bivariate(adata, resource_name="consensus", local_name="cosine", global_name=None, n_perms=None)
    >>> li.ms.nmf(lrdata, n_components=3, random_state=0)
    >>> loadings = li.ms.get_variable_loadings(lrdata, varm_key="NMF_H", variable_sep="^")

    The separators split those composite names back into their parts, and the rows
    are ordered by the absolute loading on the first factor:

    >>> loadings.head(3).round(3)
      ligand_complex receptor_complex  Factor1  Factor2  Factor3
    6       HLA-DPB1              CD4    2.518      0.0    0.114
    4       HLA-DQB1              CD4    2.498      0.0    0.034
    0        HLA-DRA              CD4    2.486      0.0    0.160

    Views built by :func:`liana.ms.lrs_to_views` name their variables
    `'source&target:ligand^receptor'`, which `view_sep=':'`, `variable_sep='^'` and
    `pair_sep='&'` split in the same way.
    """
    if var_names is None:
        var_names = ["ligand_complex", "receptor_complex"]
    if pair_names is None:
        pair_names = ["source", "target"]

    if loadings is not None:
        # loadings supplied directly (e.g. from a MOFA-Flex model's get_weights());
        # a dict of per-view {view: features x factors} is concatenated feature-wise
        frame = pd.concat(list(loadings.values()), axis=0) if isinstance(loadings, dict) else loadings
        df = pd.DataFrame(frame).copy()
        factor_cols = list(df.columns)
    else:
        if adata is None or varm_key is None or varm_key not in adata.varm.keys():
            raise ValueError(f"{varm_key} not found in adata.varm")
        loading_matrix = np.asarray(adata.varm[varm_key])
        factor_cols = [f"Factor{i + 1}" for i in range(loading_matrix.shape[1])]
        df = pd.DataFrame(index=get_var(adata).index, data=loading_matrix, columns=factor_cols)

    df.index.name = None
    df = df.reset_index()

    if view_sep:
        df[["view", "variable"]] = df["index"].str.split(view_sep, expand=True)

        if drop_columns:
            df.drop(columns="index", inplace=True)

    if variable_sep:
        if view_sep is None:
            df = df.rename(columns={"index": "variable"})

        df[var_names] = df["variable"].str.split(variable_sep, expand=True)

        if drop_columns:
            df.drop(columns="variable", inplace=True)

    if pair_sep:
        df[pair_names] = df["view"].str.split(pair_sep, expand=True)

        if drop_columns:
            df.drop(columns="view", inplace=True)

    # Re-order columns so that factors are last
    df = df.reindex(sorted(df.columns, key=lambda x: x.startswith("Factor")), axis=1)

    # re-order to absolute values
    df = df.reindex(df[factor_cols[0]].abs().sort_values(ascending=False).index)

    return df
