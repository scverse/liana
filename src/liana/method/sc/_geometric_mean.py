import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.stats import gmean

from liana._core._pipe_utils._get_mean_perms import _apply_proximity_weights, _calculate_pvals
from liana.method.sc._Method import Method, MethodMeta


def _gmean_score(
    x: DataFrame,
    perm_stats: NDArray[np.floating] | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    """
    Calculate CellPhoneDB-like LR means and p-values

    Parameters
    ----------
    x
        DataFrame with LIANA results
    perm_stats
        Permutation statistics with shape (2 (ligand-receptor), n_perms (number of permutations), n_rows (in lr_res)

    Returns
    -------
    A tuple with lr_mean and p-value for x
    """
    lr_gmeans = np.asarray(gmean((x["ligand_means"].to_numpy(), x["receptor_means"].to_numpy()), axis=0))
    weighted = _apply_proximity_weights(lr_gmeans, x)

    gmean_pvals = _calculate_pvals(weighted, perm_stats, gmean)

    return weighted, gmean_pvals


_geometric_mean = MethodMeta(
    method_name="Geometric Mean",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=[],
    fun=_gmean_score,
    magnitude="lr_gmeans",
    magnitude_ascending=False,
    specificity="gmean_pvals",
    specificity_ascending=True,
    permute=True,
    reference="CellPhoneDBv2's permutation approach applied to the "
    "geometric means of ligand-receptors' mean, as opposed to "
    "their arithmetic mean.",
)

geometric_mean = Method(_method=_geometric_mean)
