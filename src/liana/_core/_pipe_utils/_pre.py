from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame, Index
from scipy.sparse import csr_matrix, isspmatrix_csr

from liana._core._common import _logg
from liana._core._types import _to_matrix, get_obs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike, NDArray

    from liana._core._types import ObsmValue


def assert_covered(
    subset: ArrayLike,
    superset: ArrayLike,
    subset_name: str = "resource",
    superset_name: str = "var_names",
    prop_missing_allowed: float = 0.98,
    verbose: bool = False,
) -> None:
    """
    Assert if elements are covered at a decent proportion

    Parameters
    ----------
    subset
        Subset of elements
    superset
        The superset of elements
    subset_name
        Name of the subset
    superset_name
        Name of the superset
    prop_missing_allowed
        Allowed proportion of missing/mismatched elements in the subset
    verbose
        Verbosity flag

    Raises
    ------
    ValueError
        When the the number of missing elements in subset (with respect to superset) is over the threshold
    """
    subset_arr = np.asarray(subset)
    is_missing = ~np.isin(subset_arr, superset)
    if subset_arr.size == 0:
        prop_missing = 1.0
        x_missing = "values in interactions argument"
    else:
        prop_missing = float(np.sum(is_missing) / subset_arr.size)
        x_missing = ", ".join(str(entity) for entity in subset_arr[is_missing])
    if prop_missing > prop_missing_allowed:
        msg = (
            f"Please check if appropriate organism/ID type was provided! "
            f"Allowed proportion ({prop_missing_allowed}) of missing "
            f"{subset_name} elements exceeded ({prop_missing:.2f}). "
            f"Too few features from the resource were found in the data."
        )
        raise ValueError(msg + f" [{x_missing}] missing from {superset_name}")

    _logg(
        f"{prop_missing:.2f} of entities in the resource are missing from the data.",
        verbose=verbose & (prop_missing > 0),
    )


def prep_check_adata(
    adata: AnnData,
    groupby: str | None,
    min_cells: int | None,
    groupby_subset: ArrayLike | None = None,
    use_raw: bool = False,
    layer: str | None = None,
    obsm: Mapping[str, ObsmValue | AnnData] | None = None,
    uns: dict[str, Any] | None = None,
    complex_sep: str | None = "_",
    verbose: bool = False,
) -> AnnData:
    """
    Check if the anndata object is in the correct format and preprocess

    Parameters
    ----------
    adata
        Un-formatted Anndata.
    groupby
        Column to groupby. None if the ligand-receptor pipe
        calling this function does not rely on cell labels.
        For example, if ligand-receptor stats are needed
        for the whole sample (global).
    min_cells
        minimum cells per cell identity. None if groupby is not passed.
    groupby_subset
        Collection of `obs` names, if provided, subsets the subgroups from groupby
    use_raw
        Use raw attribute of adata if present.
    layer
        Indicate whether to use any layer.
    obsm
        `AnnData.obsm` matrix collection to include in the resulting AnnData
    uns
        `AnnData.uns` unspecified mappings to inmclude in the resulting AnnData
    complex_sep
        Separator to use for complex names.
    verbose
        Verbosity flag.

    Raises
    ------
    ValueError
        If the data matrix contains non-finite values (NaN or Inf)

    Returns
    -------
    Anndata object to be used downstream
    """
    X = _choose_mtx_rep(adata=adata, use_raw=use_raw, layer=layer, verbose=verbose)
    old_obsp = dict(adata.obsp)

    if use_raw and layer is None:
        if adata.raw is None:
            raise ValueError("`.raw` is not initialized!")
        var = DataFrame(index=adata.raw.var_names)
    else:
        var = DataFrame(index=adata.var_names)

    # discard any instances of AnnData if in obsm
    obsm_arrays: dict[str, ObsmValue] = {}
    for key, value in (obsm or {}).items():
        if not isinstance(value, AnnData):
            obsm_arrays[key] = value

    # kept as a local because `adata.X` re-widens to everything anndata can store
    X = X.astype(np.float32, copy=True)
    adata = sc.AnnData(
        X=X,
        obs=get_obs(adata).copy(),
        var=var,
        uns=uns,
    )
    # assigned rather than passed: the constructor types these as `Sequence[Any]`
    for key, pairwise in old_obsp.items():
        adata.obsp[key] = pairwise
    for key, embedding in obsm_arrays.items():
        adata.obsm[key] = embedding
    adata.var_names_make_unique()

    # Check for empty features
    msk_features = np.asarray(X.sum(axis=0)).ravel() == 0
    n_empty_features = int(np.sum(msk_features))
    if n_empty_features > 0:
        _logg(f"{n_empty_features} features of mat are empty, they will be removed.", level="warn", verbose=verbose)
        adata = adata[:, ~msk_features]
        X = X[:, ~msk_features]

    # Check for empty samples
    msk_samples = np.asarray(X.sum(axis=1)).ravel() == 0
    n_empty_samples = int(np.sum(msk_samples))
    if n_empty_samples > 0:
        _logg(f"{n_empty_samples} samples of mat are empty, they will be removed.", level="warn", verbose=verbose)

    # Check if log-norm
    _sum = np.sum(X.data[0:100])
    if _sum == np.floor(_sum):
        _logg("Make sure that normalized counts are passed!", level="warn", verbose=verbose)

    # Check for non-finite values
    if np.any(~np.isfinite(X.data)):
        raise ValueError("mat contains non finite values (nan or inf), please set them to 0 or remove them.")

    if groupby is not None:
        _check_groupby(adata, groupby, verbose)

        if groupby_subset is not None:
            adata = adata[get_obs(adata)[groupby].isin(np.asarray(groupby_subset)), :]

        obs = get_obs(adata)
        obs["@label"] = obs[groupby]

        # Remove any cell types below X number of cells per cell type
        count_cells = obs.groupby(groupby)[groupby].size().reset_index(name="count").copy()
        count_cells["keep"] = count_cells["count"] >= min_cells

        if not all(count_cells.keep):
            lowly_abundant_idents = list(count_cells[~count_cells.keep][groupby])
            # remove lowly abundant identities
            msk = ~np.isin(obs[[groupby]], lowly_abundant_idents)
            adata = adata[msk]
            _logg(
                "The following cell identities were excluded: {}".format(", ".join(lowly_abundant_idents)),
                level="warn",
                verbose=verbose,
            )

    check_vars(adata.var_names, complex_sep=complex_sep, verbose=verbose)
    # Re-order adata vars alphabetically
    adata = adata[:, np.sort(adata.var_names)]
    return adata


def check_vars(var_names: Iterable[str], complex_sep: str | None, verbose: bool = False) -> list[str]:
    """
    Raise a warning if `complex_sep` is part of any variable name.

    Parameters
    ----------
    var_names
        Variable names to check
    complex_sep
        Separator or any substring to check for in the variable names
    %(verbose)s
    """
    var_issues = [] if complex_sep is None else [name for name in var_names if complex_sep in name]

    _logg(
        f"{var_issues} contain `{complex_sep}`. Consider replacing those!",
        verbose=verbose & (len(var_issues) > 0),
        level="warn",
    )

    return var_issues


def filter_resource(resource: DataFrame, var_names: Index | NDArray[Any]) -> DataFrame:
    """
    Filter interactions for which vars are not present.

    Note that here I remove any interaction that /w genes that are not found
    in the dataset. Note that this is not necessarily the case in liana-r.
    There, I assign the expression of those with missing subunits to 0, while
    those without any subunit present are implicitly filtered.

    Parameters
    ----------
    resource
        Resource with 'ligand' and 'receptor' columns
    var_names
        Relevant variables - i.e. the variables to be used downstream

    Returns
    -------
    A filtered resource DataFrame
    """
    # Remove those without any subunit
    resource = resource[(np.isin(resource.ligand, var_names)) & (np.isin(resource.receptor, var_names))]

    # Only keep interactions /w complexes for which all subunits are present
    missing_comps = resource[resource.interaction.str.contains("_")].copy()
    missing_comps["all_units"] = missing_comps["ligand_complex"] + "_" + missing_comps["receptor_complex"]

    # Get those not with all subunits
    missing_comps = missing_comps[
        np.logical_not([all(x in var_names for x in entity.split("_")) for entity in missing_comps.all_units])
    ]
    # Filter them
    return resource[~resource.interaction.isin(missing_comps.interaction)]


def _choose_mtx_rep(
    adata: AnnData, use_raw: bool = False, layer: str | None = None, verbose: bool = False
) -> csr_matrix:
    """
    Choose matrix (adapted from scanpy)

    Parameters
    ----------
    adata
        Unformatted Anndata.
    use_raw
        Use raw attribute of adata if present.
    layer
        Indicate whether to use any layer.

    Returns
    -------
        The matrix to be used by LIANA+.
    """
    if layer is not None and use_raw:
        raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
    if layer is not None:
        _logg(f"Using the `{layer}` layer!", verbose=verbose)
        chosen = _to_matrix(adata.layers[layer], what=f"adata.layers[{layer!r}]")
    elif use_raw:
        if adata.raw is None:
            raise ValueError("`.raw` is not initialized!")
        _logg("Using `.raw`!", verbose=verbose)
        chosen = _to_matrix(adata.raw.X, what="adata.raw.X")
    else:
        _logg("Using `.X`!", verbose=verbose)
        chosen = _to_matrix(adata.X, what="adata.X")

    # convert to sparse csr matrix
    if isspmatrix_csr(chosen):
        return chosen
    _logg("Converting to sparse csr matrix!", verbose=verbose)
    return csr_matrix(chosen)


def _check_groupby(adata: AnnData, groupby: str, verbose: bool) -> None:
    obs = get_obs(adata)
    if groupby not in obs.columns:
        raise KeyError(f"`{groupby}` not found in `adata.obs.columns`.")
    if not obs[groupby].dtype.name == "category":
        _logg(f"Converting `{groupby}` to categorical!", level="warn", verbose=verbose)
        obs[groupby] = obs[groupby].astype("category")
