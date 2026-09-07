import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, diags
from tqdm import tqdm

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._docs import d
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._types import get_obs


def _get_group_mean(
    X: csr_matrix,
    groupby_labels: pd.Series | np.ndarray,
    var_names: pd.Index,
    groups_order: list[str] | None = None,
) -> pd.DataFrame:
    s = pd.Series(groupby_labels)
    groups_dum = pd.get_dummies(s, dummy_na=False)

    dummies = csr_matrix(groups_dum)
    col_sums = np.asarray(dummies.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    scaled = dummies @ diags(1.0 / col_sums)
    result = scaled.T @ X
    arr = result.toarray()
    df_raw = pd.DataFrame(arr, index=groups_dum.columns, columns=var_names)

    return df_raw


@d.dedent
def compute_global_specificity(
    adata: AnnData,
    groupby: str,
    lr_sep: str | None = V.lr_sep,
    n_perms: int = V.n_perms,
    seed: int = V.seed,
    n_jobs: int = -1,
    verbose: bool = V.verbose,
    use_raw: bool = V.use_raw,
    layer: str | None = V.layer,
    uns_key: str = "global_interactions",
) -> None:
    """
    Computes group-specific ligand-receptor means and permutation-based p-values.

    This function performs a one-sided permutation test to assess whether the observed
    group-specific means are significantly higher than expected by chance. P-values are
    computed using the formula (k + 1) / (n_perms + 1), where k is the number of
    permutations with values greater than or equal to the observed statistic.

    Parameters
    ----------
    %(adata)s
    %(groupby)s
    %(lr_sep)s
    %(n_perms)s
    %(seed)s
    n_jobs
        Number of parallel jobs. Defaults to -1 (all available cores).
    %(verbose)s
    %(use_raw)s
    %(layer)s
    %(uns_key)s

    Returns
    -------
    None. The result with 'lr_mean' and 'pval' is stored in `adata.uns["global_interactions"]`.

    Examples
    --------
    Takes the output of ``liana.mt.inflow``, whose `var_names` are the
    `'source^ligand^receptor'` triplets that `lr_sep` splits back apart:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> lrdata = li.mt.inflow(adata, groupby="bulk_labels", resource_name="consensus")
    >>> li.mt.compute_global_specificity(lrdata, groupby="bulk_labels", n_perms=10, use_raw=False, n_jobs=1)

    One row per sender, ligand, receptor and receiver group, sorted by p-value. Use many
    more permutations than the 10 here -- the smallest attainable p-value is
    `1/(n_perms+1)`. `n_jobs=1` avoids the process-spawn overhead that the default `-1`
    costs at this size.
    """
    if groupby not in adata.obs.columns:
        raise KeyError(f"`groupby`='{groupby}' not found in adata.obs.")

    X = _choose_mtx_rep(adata, layer=layer, use_raw=use_raw)
    var_names = adata.var_names
    original_groupby_labels = get_obs(adata)[groupby].astype("category")
    groups_order = list(original_groupby_labels.cat.categories)

    # Compute observed statistic
    df_obs = _get_group_mean(X, original_groupby_labels, var_names, groups_order=groups_order)
    obs_values = df_obs.values

    _logg(f"Running {n_perms} permutations in parallel...", verbose=verbose)

    joblib_verbose = 5 if verbose else 0

    rng_main = np.random.default_rng(seed)
    perm_stats_list = Parallel(n_jobs=n_jobs, verbose=joblib_verbose)(
        delayed(_get_group_mean)(
            X, rng_main.permutation(original_groupby_labels.to_numpy()), var_names, groups_order=groups_order
        )
        for _ in tqdm(range(n_perms), desc="Running permutations", disable=not verbose)
    )

    # Convert results to array
    perm_stats = np.array([df_perm.values for df_perm in perm_stats_list])

    # Compute empirical p-values
    k = np.sum(perm_stats >= obs_values, axis=0)
    p_values = (k + 1) / (n_perms + 1)
    pval_df = pd.DataFrame(p_values, index=df_obs.index, columns=df_obs.columns)
    pval_melted = pval_df.reset_index().melt(id_vars="index", var_name="feature", value_name="pval")

    # Final df
    df_melted = df_obs.reset_index().melt(id_vars="index", var_name="feature", value_name="lr_mean")
    merged_df = df_melted.merge(pval_melted[["index", "feature", "pval"]], on=["index", "feature"], how="left")

    if lr_sep is not None:
        merged_df[["source", "ligand_complex", "receptor_complex"]] = merged_df["feature"].str.split(
            lr_sep, expand=True
        )
        merged_df = merged_df.rename(columns={"index": "target"})
        final_df = merged_df[["source", "ligand_complex", "receptor_complex", "target", "lr_mean", "pval"]]
    else:
        final_df = merged_df

    final_df = final_df.sort_values("pval", ascending=True)

    # Save result
    adata.uns[uns_key] = final_df
