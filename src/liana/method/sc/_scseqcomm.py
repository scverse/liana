import numpy as np
from pandas import DataFrame, Series

from liana.method.sc._Method import Method, MethodMeta


def _inter_score(x: DataFrame) -> tuple[Series, None]:
    """
    Computes the element-wise minimum between pairs of ligand-receptor CDF

    Parameters
    ----------
    x
        DataFrame containing the ligand and receptor CDF

    Returns
    -------
    inter_score
        Pair-wise minimum CDF
    None
    """
    inter_score = np.minimum(x["ligand_cdf"], x["receptor_cdf"])
    return Series(inter_score, index=x.index), None


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
