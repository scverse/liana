from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData
from mudata import MuData
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix, hstack, isspmatrix_csr

from liana._core._constants import DefaultValues as V
from liana._core._pipe_utils import prep_check_adata
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._types import copy_aligned, get_obs, get_var
from liana.multisample.mdata_to_anndata import mdata_to_anndata

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _add_complexes_to_var(
    adata: AnnData,
    entities: NDArray[Any],
    complex_sep: str = "_",
) -> AnnData:
    """Generate an AnnData object with complexes appended as variables."""
    complexes = entities[Series(entities).str.contains(complex_sep)]
    var = get_var(adata)
    X: csr_matrix | None = None
    for comp in complexes:
        subunits = comp.split(complex_sep)

        # keep only complexes, the subunits of which are in var
        if all(subunit in var.index for subunit in subunits):
            var.loc[comp, :] = None

            # create matrix for this complex
            new_array = csr_matrix(_choose_mtx_rep(adata[:, subunits]).min(axis=1))

            X = new_array if X is None else csr_matrix(hstack((X, new_array)))

    if X is None:  # no complex had all of its subunits present
        return adata

    combined = AnnData(
        X=csr_matrix(hstack((_choose_mtx_rep(adata), X))),
        obs=get_obs(adata),
        var=var,
        uns=dict(adata.uns),
    )
    copy_aligned(combined, obsm=adata.obsm, obsp=adata.obsp, varm=adata.varm)
    return combined


# the x/y feature matrices reach here straight off `.X`, so still sparse; `mat.mean(axis)`
# returns a dense matrix, so everything downstream of the centring is dense either way
type _ZScoreInput = np.ndarray | csr_matrix


def _zscore(mat: _ZScoreInput, axis: int = 0, global_r: bool = False) -> np.ndarray:
    # NOTE: `global_r=True` is specific to global SpatialDM
    spot_n = 1 if global_r else mat.shape[axis]

    centred = mat - mat.mean(axis=axis)
    scaled = centred / np.sqrt(np.sum(np.power(centred, 2), axis=axis) / spot_n)

    return np.array(np.clip(scaled, -10, 10))


def _spatialdm_weight_norm(weight: np.ndarray | csr_matrix) -> np.ndarray | csr_matrix:
    norm_factor = weight.shape[0] / weight.sum()
    normalised: np.ndarray | csr_matrix = norm_factor * weight
    return normalised


def _rename_means(lr_stats: DataFrame, entity: str) -> DataFrame:
    df = lr_stats.copy()
    df.columns = df.columns.map(lambda x: entity + "_" + str(x) if x != "gene" else "gene")
    return df.rename(columns={"gene": entity})


def _validate_kwargs(expected_params: set[str], **kwargs: object) -> None:
    unexpected_kwargs = set(kwargs) - expected_params
    if unexpected_kwargs:
        raise ValueError(f"Unexpected keyword arguments: {unexpected_kwargs}")


def _handle_connectivity(adata: AnnData, connectivity_key: str) -> csr_matrix:
    """
    Extract and validate spatial connectivity matrix from AnnData.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing connectivity in obsp.
    connectivity_key : str
        Key in adata.obsp to retrieve connectivity matrix.

    Returns
    -------
    connectivity : csr_matrix
        Sparse connectivity matrix in CSR format with float32 dtype.
    """
    if connectivity_key not in adata.obsp.keys():
        raise ValueError(f"No connectivity matrix found in adata.obsp[{connectivity_key}]")
    connectivity = adata.obsp[connectivity_key]

    if not isspmatrix_csr(connectivity) or (connectivity.dtype != np.float32):
        connectivity = csr_matrix(connectivity, dtype=np.float32)

    return connectivity


def _check_instance(mdata: AnnData | MuData) -> bool:
    """
    Check if input is MuData or AnnData.

    Parameters
    ----------
    mdata : MuData | AnnData
        Input data object.

    Returns
    -------
    is_mudata : bool
        True if MuData, False if AnnData.

    Raises
    ------
    ValueError
        If input is neither MuData nor AnnData.
    """
    if isinstance(mdata, MuData):
        return True
    elif isinstance(mdata, AnnData):
        return False
    else:
        raise ValueError("Input must be an AnnData or MuData object")


def _process_anndata(
    adata: AnnData,
    complex_sep: str | None,
    verbose: bool | None,
    **kwargs: Any,
) -> tuple[AnnData, str, str]:
    """
    Process AnnData input for spatial methods.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    complex_sep : str | None
        Separator for complex names.
    verbose : bool
        Verbosity flag.
    **kwargs : dict
        Optional parameters: 'x_name', 'y_name', 'use_raw', 'layer'.

    Returns
    -------
    adata : AnnData
        Processed AnnData object.
    x_name : str
        Name for x features (defaults to 'ligand' if not provided in kwargs).
    y_name : str
        Name for y features (defaults to 'receptor' if not provided in kwargs).

    Notes
    -----
    For backward compatibility, defaults to 'ligand'/'receptor' for AnnData.
    """
    expected_params = {"x_name", "y_name", "use_raw", "layer"}
    _validate_kwargs(expected_params=expected_params, **kwargs)

    # For backward compatibility: default to 'ligand'/'receptor' for AnnData
    x_name = kwargs.get("x_name", "ligand")
    y_name = kwargs.get("y_name", "receptor")

    adata = prep_check_adata(
        adata=adata,
        use_raw=kwargs.get("use_raw", V.use_raw),
        layer=kwargs.get("layer", V.layer),
        verbose=verbose,
        obsm=adata.obsm.copy(),
        uns=dict(adata.uns),
        groupby=None,
        min_cells=None,
        complex_sep=complex_sep,
    )

    return adata, x_name, y_name


def _process_mudata(
    mdata: MuData,
    complex_sep: str | None,
    verbose: bool | None,
    **kwargs: Any,
) -> tuple[AnnData, str, str]:
    """
    Process MuData input for spatial methods.

    Parameters
    ----------
    mdata : MuData
        Input MuData object.
    complex_sep : str | None
        Separator for complex names.
    verbose : bool
        Verbosity flag.
    **kwargs : dict
        Required parameters: 'x_mod', 'y_mod'
        Optional parameters: 'x_name', 'y_name', 'x_use_raw', 'y_use_raw',
        'x_layer', 'y_layer', 'x_transform', 'y_transform'.

    Returns
    -------
    adata : AnnData
        Converted and processed AnnData object.
    x_name : str
        Name for x features (defaults to 'x' if not provided in kwargs).
    y_name : str
        Name for y features (defaults to 'y' if not provided in kwargs).
    """
    expected_params = {
        "x_name",
        "y_name",
        "x_mod",
        "y_mod",
        "x_use_raw",
        "x_layer",
        "y_use_raw",
        "y_layer",
        "x_transform",
        "y_transform",
    }
    _validate_kwargs(expected_params=expected_params, **kwargs)

    x_mod = kwargs.get("x_mod")
    y_mod = kwargs.get("y_mod")

    if x_mod is None or y_mod is None:
        raise ValueError("MuData processing requires 'x_mod' and 'y_mod' parameters.")

    # For MuData: default to 'x'/'y' if not explicitly provided in kwargs
    x_name = kwargs.get("x_name", "x")
    y_name = kwargs.get("y_name", "y")

    adata = mdata_to_anndata(
        mdata,
        x_mod=x_mod,
        y_mod=y_mod,
        x_use_raw=kwargs.get("x_use_raw", V.use_raw),
        x_layer=kwargs.get("x_layer", V.layer),
        y_use_raw=kwargs.get("y_use_raw", V.use_raw),
        y_layer=kwargs.get("y_layer", V.layer),
        x_transform=kwargs.get("x_transform"),
        y_transform=kwargs.get("y_transform"),
        verbose=verbose,
    )

    adata = prep_check_adata(
        adata=adata,
        use_raw=False,
        layer=None,
        verbose=verbose,
        obsm=adata.obsm.copy(),
        uns=dict(adata.uns),
        groupby=None,
        min_cells=None,
        complex_sep=complex_sep,
    )

    return adata, x_name, y_name
