from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from liana.method.sc._Method import Method, MethodMeta


def _gmean_score(x: Mapping[str, NDArray[np.floating]]) -> tuple[NDArray[np.floating], None]:
    """Geometric Mean: `lr_gmeans` = geometric mean of the ligand and receptor means.

    The same arithmetic as `scipy.stats.gmean` (bit-identical), without stacking the inputs into a new array.
    """
    ligand, receptor = x["ligand_means"], x["receptor_means"]
    dtype = np.result_type(ligand, receptor, 1.0)  # promote before the log, as `scipy.stats.gmean` does
    with np.errstate(divide="ignore"):  # log(0) = -inf -> exp(-inf) = 0, as in `scipy.stats.gmean`
        log_sum = np.log(ligand.astype(dtype, copy=False)) + np.log(receptor.astype(dtype, copy=False))
    return np.exp(log_sum / 2), None


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
