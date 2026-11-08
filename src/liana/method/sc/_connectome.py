from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from liana.method.sc._Method import Method, MethodMeta


def _connectome_score(x: Mapping[str, NDArray[np.floating]]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Connectome: `expr_prod` = ligand-receptor product; `scaled_weight` = mean of their z-scores."""
    return x["ligand_means"] * x["receptor_means"], (x["ligand_zscores"] + x["receptor_zscores"]) / 2


_connectome = MethodMeta(
    method_name="Connectome",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=["ligand_zscores", "receptor_zscores"],
    fun=_connectome_score,
    magnitude="expr_prod",
    magnitude_ascending=False,
    specificity="scaled_weight",
    specificity_ascending=False,
    permute=False,
    reference="Raredon, M.S.B., Yang, J., Garritano, J., Wang, M., Kushnir, "
    "D., Schupp, J.C., Adams, T.S., Greaney, A.M., Leiby, K.L., "
    "Kaminski, N. and Kluger, Y., 2022. Computation and "
    "visualization of cell–cell signaling topologies in "
    "single-cell systems data using Connectome. Scientific "
    "reports, 12(1), pp.1-12. ",
)

connectome = Method(_method=_connectome)
