from pandas import DataFrame, Series

from liana._core._common import _logg
from liana.method.sc._Method import Method, MethodMeta


def _spec_weight(
    ligand_means: Series,
    ligand_means_sums: Series,
    receptor_means: Series,
    receptor_means_sums: Series,
) -> Series:
    s_lig = ligand_means / ligand_means_sums
    s_rec = receptor_means / receptor_means_sums
    return s_lig * s_rec


def _natmi_score(x: DataFrame) -> tuple[Series, Series]:
    # magnitude
    expr_prod = x["ligand_means"] * x["receptor_means"]

    # specificity
    spec_weight = _spec_weight(x["ligand_means"], x["ligand_means_sums"], x["receptor_means"], x["receptor_means_sums"])
    # the normaliser sums over the labels in the run, so a single source or target leaves nothing to normalise against
    for what in ("ligand", "receptor"):
        if x[f"{what}_means"].eq(x[f"{what}_means_sums"]).all():
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
