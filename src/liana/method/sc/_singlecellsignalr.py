import numpy as np
from numpy.typing import NDArray

from liana.method.sc._Method import Method, MethodMeta


def _sca_score(
    ligand: NDArray[np.floating],
    receptor: NDArray[np.floating],
    *,
    mat_mean: NDArray[np.floating],
) -> tuple[NDArray[np.floating], None]:
    """SingleCellSignalR-like LRscore."""
    lr_sqrt = np.sqrt(ligand) * np.sqrt(receptor)
    return lr_sqrt / (lr_sqrt + mat_mean), None


# Initialize CPDB Meta
_singlecellsignalr = MethodMeta(
    method_name="SingleCellSignalR",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=["mat_mean"],
    fun=_sca_score,
    magnitude="lrscore",
    magnitude_ascending=False,
    specificity=None,
    specificity_ascending=None,
    permute=False,
    reference="Cabello-Aguilar, S., Alame, M., Kon-Sun-Tack, F., Fau, "
    "C., Lacroix, M. and Colinge, J., "
    "2020. SingleCellSignalR: inference of intercellular "
    "networks from single-cell transcriptomics. Nucleic "
    "Acids Research, 48(10), pp.e55-e55. ",
)

singlecellsignalr = Method(_method=_singlecellsignalr)
