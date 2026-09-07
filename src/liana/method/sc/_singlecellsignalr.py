import numpy as np
from pandas import DataFrame, Series

from liana.method.sc._Method import Method, MethodMeta


def _sca_score(x: DataFrame) -> tuple[Series, None]:
    """
    Calculate SingleCellSignalR-like LRscore

    Parameters
    ----------
    x
        DataFrame containing the ligand and receptor means

    Returns
    -------
    LRscore
        The ligand-receptor scores
    None
    """
    lr_sqrt = np.sqrt(x["ligand_means"]) * np.sqrt(x["receptor_means"])
    denominator = lr_sqrt + x.mat_mean
    return lr_sqrt / denominator, None


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
