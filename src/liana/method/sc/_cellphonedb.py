import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from liana.method.sc._Method import Method, MethodMeta


def _mean(ligand: NDArray[np.floating], receptor: NDArray[np.floating]) -> NDArray[np.floating]:
    """CellPhoneDB's interaction score: the mean of the ligand and receptor cluster means."""
    return np.asarray(np.mean((ligand, receptor), axis=0))


def _zero_mask(observed: NDArray[np.floating], x: DataFrame) -> NDArray[np.floating]:
    """Score an interaction 0 when its ligand or receptor is not expressed at all.

    Applied to the observed score only; a 0 is reached by every permutation, so its p-value is 1.
    """
    observed[((x["ligand_means"] == 0) | (x["receptor_means"] == 0)).to_numpy()] = 0
    return observed


# Initialize CPDB Meta
_cellphonedb = MethodMeta(
    method_name="CellPhoneDB",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=[],
    fun=None,
    combine=_mean,
    observed_fn=_zero_mask,
    magnitude="lr_means",
    magnitude_ascending=False,
    specificity="cellphone_pvals",
    specificity_ascending=True,
    permute=True,
    reference="Efremova, M., Vento-Tormo, M., Teichmann, S.A. and "
    "Vento-Tormo, R., 2020. CellPhoneDB: inferring cell–cell "
    "communication from combined expression of multi-subunit "
    "ligand–receptor complexes. Nature protocols, 15(4), "
    "pp.1484-1506. ",
)

cellphonedb = Method(_method=_cellphonedb)
