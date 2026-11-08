import numpy as np
from numpy.typing import NDArray

from liana.method.sc._Method import Method, MethodMeta


def _inter_score(
    ligand: NDArray[np.floating],
    receptor: NDArray[np.floating],
    *,
    ligand_cdf: NDArray[np.floating],
    receptor_cdf: NDArray[np.floating],
) -> tuple[NDArray[np.floating], None]:
    """ScSeqComm's inter score: the elementwise minimum of the ligand and receptor CDFs; their means are not used."""
    return np.minimum(ligand_cdf, receptor_cdf), None


_scseqcomm = MethodMeta(
    method_name="scSeqComm",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=["ligand_cdf", "receptor_cdf"],
    fun=_inter_score,
    magnitude="inter_score",
    magnitude_ascending=False,
    specificity=None,
    specificity_ascending=None,
    permute=False,
    reference="Baruzzo, G., Cesaro, G., Di Camillo, B. "
    "2022. Identify, quantify and characterize cellular communication "
    "from single-cell RNA-sequencing data with scSeqComm. Bioinformatics, "
    "38(7), pp.1920-1929",
)
scseqcomm = Method(_method=_scseqcomm)
