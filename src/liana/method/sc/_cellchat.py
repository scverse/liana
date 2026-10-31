import numpy as np
from numpy.typing import NDArray

from liana.method.sc._Method import Method, MethodMeta

_KH = 0.5
"""Half-maximal constant of CellChat's Hill function."""


# simplified/resource-generalizable cellchat probability score
def _lr_probability(ligand: NDArray[np.floating], receptor: NDArray[np.floating]) -> NDArray[np.floating]:
    """CellChat's interaction probability: a Hill function of the ligand-receptor trimean product."""
    lr_prob = np.prod((ligand, receptor), axis=0)
    return np.asarray(lr_prob / (_KH + lr_prob))


# Initialize CellChat Meta
_cellchat = MethodMeta(
    method_name="CellChat",
    complex_cols=["ligand_trimean", "receptor_trimean"],
    add_cols=["mat_max"],
    fun=None,
    combine=_lr_probability,
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
