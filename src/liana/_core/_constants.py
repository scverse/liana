from __future__ import annotations

from typing import Final, Literal

from numpy import exp, finfo, floating, log10
from numpy.typing import NDArray
from pandas import Series

type DeMethod = Literal["logreg", "t-test", "wilcoxon", "t-test_overestim_var"]
"""The differential-expression tests :func:`scanpy.tl.rank_genes_groups` supports."""


class DefaultValues:
    """Default Values"""

    logbase: Final = exp(1)
    min_cells: Final = 5
    expr_prop: Final = 0.05
    nz_prop: Final = 0.05
    n_perms: Final = 1000
    seed: Final = 1337
    de_method: Final[DeMethod] = "t-test"
    resource_name: Final = "consensus"
    resource: Final[None] = None
    interactions: Final[None] = None
    layer: Final[None] = None
    use_raw: Final = False
    verbose: Final = False
    return_all_lrs: Final = False
    supp_columns: Final[None] = None
    inplace: Final = True
    groupby_pairs: Final[None] = None

    return_fig: Final = True
    cmap: Final = "viridis"

    lr_sep: Final = "^"
    complex_sep: Final = "_"

    @staticmethod
    def inverse_fn[T: (Series, NDArray[floating])](x: T) -> T:
        """Turn a "lower is stronger" score into a "higher is stronger" one.

        Called both with a DataFrame column (:func:`liana.method.process_scores`, the plotting modules) and with a bare array, and returns whichever it was given.
        """
        if isinstance(x, Series):
            # numpy's ufunc overloads return `Any` for a Series
            return Series(-log10(x.to_numpy() + finfo(float).eps), index=x.index, name=x.name)
        return -log10(x + finfo(float).eps)


class Keys:
    """Keys related to AnnData"""

    uns_key: Final = "liana_res"
    spatial_key: Final = "spatial"
    connectivity_key: Final = f"{spatial_key}_connectivities"
    target_metrics: Final = "target_metrics"
    interactions: Final = "interactions"


class PrimaryColumns:
    source: Final = "source"
    target: Final = "target"
    ligand: Final = "ligand"
    receptor: Final = "receptor"
    ligand_complex: Final = "ligand_complex"
    receptor_complex: Final = "receptor_complex"
    primary: Final[list[str]] = [source, target, ligand_complex, receptor_complex]
    complete: Final[list[str]] = primary + [ligand, receptor]


class CommonColumns:
    ligand_means: Final = "ligand_means"
    receptor_means: Final = "receptor_means"
    ligand_props: Final = "ligand_props"
    receptor_props: Final = "receptor_props"
    ligand_pvals: Final = "ligand_pvals"
    receptor_pvals: Final = "receptor_pvals"
    proximity: Final = "proximity"


class MethodColumns:
    ligand_means_sums: Final = "ligand_means_sums"
    receptor_means_sums: Final = "receptor_means_sums"
    ligand_zscores: Final = "ligand_zscores"
    receptor_zscores: Final = "receptor_zscores"
    ligand_logfc: Final = "ligand_logfc"
    receptor_logfc: Final = "receptor_logfc"
    ligand_trimean: Final = "ligand_trimean"
    receptor_trimean: Final = "receptor_trimean"
    mat_mean: Final = "mat_mean"
    mat_max: Final = "mat_max"
    ligand_cdf: Final = "ligand_cdf"
    receptor_cdf: Final = "receptor_cdf"

    @classmethod
    def get_all_values(cls) -> list[str]:
        return [value for name, value in cls.__dict__.items() if not name.startswith("__") and isinstance(value, str)]


class InternalValues:
    lrs_to_keep: Final = "lrs_to_keep"
    prop_min: Final = "prop_min"
    label: Final = "@label"
