from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import anndata as an

from liana._core._docs import d
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._types import MatrixLike, copy_aligned, get_x

if TYPE_CHECKING:
    from mudata import MuData

type _MatrixTransform = Callable[[MatrixLike], MatrixLike]


@d.dedent
def mdata_to_anndata(
    mdata: MuData,
    x_mod: str,
    y_mod: str,
    x_layer: str | None = None,
    y_layer: str | None = None,
    x_use_raw: bool = False,
    y_use_raw: bool = False,
    x_transform: _MatrixTransform | None = None,
    y_transform: _MatrixTransform | None = None,
    verbose: bool = True,
) -> an.AnnData:
    """
    Convert a MultiData object to an AnnData object.

    Parameters
    ----------
    mdata
        MuData object.
    x_mod
        Name of the modality to be used as x.
    y_mod
        Name of the modality to be used as y.
    x_layer
        Layer to be used for modality x.
    y_layer
        Layer to be used for modality y.
    x_use_raw
        Whether to use raw counts for modality x.
    y_use_raw
        Whether to use raw counts for modality y.
    x_transform
        Transformation function to be applied to modality x.
    y_transform
        Transformation function to be applied to modality y.
    %(verbose)s

    Returns
    -------
    An AnnData object with the two modalities concatenated.
    Information related to observations (obs, obsp, obsm) and `.uns` are copied from the original MuData object.

    Raises
    ------
    ValueError
        If `x_mod` and/or `y_mod` are not provided.

    Examples
    --------
    Joins two modalities of a `MuData` along the variable axis, which is what
    ``liana.mt.bivariate`` expects when relating one modality to another:

    >>> import liana as li
    >>> mdata = li.ds.generate_toy_mdata()
    >>> adata = li.ms.mdata_to_anndata(mdata, x_mod="adata_x", y_mod="adata_y", x_layer="scaled", y_layer="scaled")
    """
    xdata = _handle_mod(mdata, x_mod, x_use_raw, x_layer, x_transform, verbose)
    ydata = _handle_mod(mdata, y_mod, y_use_raw, y_layer, y_transform, verbose)

    adata = an.concat([xdata, ydata], axis=1, label="modality")

    adata.obs = mdata.obs.copy()
    adata.uns = dict(mdata.uns)
    copy_aligned(adata, obsm=dict(mdata.obsm), obsp=dict(mdata.obsp))

    return adata


def _handle_mod(
    mdata: MuData,
    mod: str,
    use_raw: bool,
    layer: str | None,
    transform: _MatrixTransform | None,
    verbose: bool,
) -> an.AnnData:
    if mod not in mdata.mod.keys():
        raise ValueError(f"`{mod}` is not in the mdata!")

    # NOTE, maybe instead of copying I can just create a minimal AnnData?
    modality = mdata.mod[mod]
    if not isinstance(modality, an.AnnData):
        raise TypeError(f"`{mod}` must be an AnnData modality, got {type(modality).__name__}.")
    md = modality.copy()
    if use_raw:
        if md.raw is None:
            raise ValueError(f"`{mod}` has no `.raw` to use.")
        md = md.raw.to_adata()
    else:
        md.X = _choose_mtx_rep(md, use_raw=use_raw, layer=layer, verbose=verbose)

    if transform:
        if verbose:
            print(f"Transforming {mod} using {transform.__name__}")
        md.X = transform(get_x(md))
    return md
