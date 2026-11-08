from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from liana.method.sc._Method import Method, MethodMeta


def _logfc_score(x: Mapping[str, NDArray[np.floating]]) -> tuple[None, NDArray[np.floating]]:
    """log2FC: `lr_logfc` = mean of the ligand and receptor log2FC."""
    return None, (x["ligand_logfc"] + x["receptor_logfc"]) / 2


_logfc = MethodMeta(
    method_name="log2FC",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=["ligand_logfc", "receptor_logfc"],
    fun=_logfc_score,
    magnitude=None,
    magnitude_ascending=None,
    specificity="lr_logfc",
    specificity_ascending=False,
    permute=False,
    reference="Dimitrov, D., Türei, D., Garrido-Rodriguez, M., Burmedi, P.L., "
    "Nagai, J.S., Boys, C., Ramirez Flores, R.O., Kim, H., Szalai, B., "
    "Costa, I.G. and Valdeolivas, A., 2022. Comparison of methods and "
    "resources for cell-cell communication inference from single-cell "
    "RNA-Seq data. Nature Communications, 13(1), pp.1-13. ",
)

logfc = Method(_method=_logfc)
