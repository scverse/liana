import numpy as np
from numpy.typing import NDArray

from liana.method.sc._Method import Method, MethodMeta


def _mean(ligand: NDArray[np.floating], receptor: NDArray[np.floating]) -> tuple[NDArray[np.floating], None]:
    """CellPhoneDB's interaction score: the mean of the ligand and receptor cluster means.

    As in CellPhoneDB, an interaction scores 0 when its ligand or receptor is not expressed,
    in the observed and the permuted data alike.
    """
    return np.where((ligand == 0) | (receptor == 0), 0, (ligand + receptor) / 2), None


# Initialize CPDB Meta
_cellphonedb = MethodMeta(
    method_name="CellPhoneDB",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=[],
    fun=_mean,
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
