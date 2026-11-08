import numpy as np
from numpy.typing import NDArray

from liana._core._common import _logg
from liana.method.sc._Method import Method, MethodMeta


def _natmi_score(
    ligand: NDArray[np.floating],
    receptor: NDArray[np.floating],
    *,
    ligand_means_sums: NDArray[np.floating],
    receptor_means_sums: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # magnitude
    expr_prod = ligand * receptor

    # specificity
    spec_weight = (ligand / ligand_means_sums) * (receptor / receptor_means_sums)
    # the normaliser sums over the labels in the run, so a single source or target leaves nothing to normalise against
    for what, stat, sums in (("ligand", ligand, ligand_means_sums), ("receptor", receptor, receptor_means_sums)):
        if np.all(stat == sums):
            _logg(
                f"NATMI's `spec_weight` is 1 for every {what}: only one {'source' if what == 'ligand' else 'target'} "
                "label is in the run (e.g. a single `groupby_pairs` pair), so specificity cannot be normalised.",
                level="warn",
            )

    return expr_prod, spec_weight


# Initialize CPDB Meta
_natmi = MethodMeta(
    method_name="NATMI",
    complex_cols=["ligand_means", "receptor_means"],
    add_cols=["ligand_means_sums", "receptor_means_sums"],
    fun=_natmi_score,
    magnitude="expr_prod",
    magnitude_ascending=False,
    specificity="spec_weight",
    specificity_ascending=False,
    permute=False,
    reference="Hou, R., Denisenko, E., Ong, H.T., Ramilowski, J.A. and Forrest, "
    "A.R., 2020. Predicting cell-to-cell communication networks using "
    "NATMI. Nature communications, 11(1), pp.1-11. ",
)

natmi = Method(_method=_natmi)
