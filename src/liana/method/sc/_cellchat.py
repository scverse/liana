import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from liana._core._pipe_utils._get_mean_perms import _apply_proximity_weights, _calculate_pvals
from liana.method.sc._Method import Method, MethodMeta

_KH = 0.5
"""Half-maximal constant of CellChat's Hill function."""


# simplified/resource-generalizable cellchat probability score
def _lr_probability(
    perm_stats: NDArray[np.floating] | tuple[NDArray[np.floating], ...],
    axis: int = 0,
) -> NDArray[np.floating]:
    lr_prob = np.prod(perm_stats, axis=axis)

    return np.asarray(lr_prob / (_KH + lr_prob))


# Internal Function to calculate CellPhoneDB LR_mean and p-values
def _cellchat_score(
    x: DataFrame,
    perm_stats: NDArray[np.floating] | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
    """
    Calculate CellChat-like LR means and p-values

    Parameters
    ----------
    x
        DataFrame with LIANA results
    perm_stats
        Permutation statistics (2 (ligand-receptor), n_perms (number of permutations, n_rows in lr_res)

    Returns
    -------
    A tuple with lr_mean and pvalue for x
    """
    lr_prob = _lr_probability((x["ligand_trimean"].to_numpy(), x["receptor_trimean"].to_numpy()))
    lr_prob = _apply_proximity_weights(lr_prob, x)
    cellchat_pvals = _calculate_pvals(lr_prob, perm_stats, _lr_probability)

    return lr_prob, cellchat_pvals


# Initialize CellChat Meta
_cellchat = MethodMeta(
    method_name="CellChat",
    complex_cols=["ligand_trimean", "receptor_trimean"],
    add_cols=["mat_max"],
    fun=_cellchat_score,
    magnitude="lr_probs",
    magnitude_ascending=False,
    specificity="cellchat_pvals",
    specificity_ascending=True,
    permute=True,
    reference="Jin, S., Guerrero-Juarez, C.F., Zhang, L., Chang, I., Ramos, "
    "R., Kuan, C.H., Myung, P., Plikus, M.V. and Nie, Q., "
    "2021. Inference and analysis of cell-cell communication using "
    "CellChat. Nature communications, 12(1), pp.1-20. ",
)

cellchat = Method(_method=_cellchat)
