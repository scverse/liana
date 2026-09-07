from typing import Literal

import numpy as np
from anndata import AnnData
from numpy.typing import NDArray
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix

from liana._core._constants import DefaultValues as V
from liana._core._docs import d
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._types import copy_aligned, get_obs, get_var


@d.dedent
def interpolate_adata(
    target: AnnData,
    reference: AnnData,
    spatial_key: str,
    layer: str | None = V.layer,
    use_raw: bool = V.use_raw,
    method: Literal["linear", "nearest", "cubic"] = "linear",
    fill_value: float = 0,
    verbose: bool = V.verbose,
) -> AnnData:
    """
    Interpolates spatial data from a target AnnData object to a reference AnnData object based on spatial coordinates.

    The function creates a new AnnData object where the `.X` attribute is filled with interpolated data using the specified method.

    Parameters
    ----------
    target
        The AnnData object to be interpolated.
    reference
        The AnnData object to be used as reference.
    %(spatial_key)s
    %(layer)s
    %(use_raw)s
    method
        Interpolation method. See `scipy.interpolate.griddata` for more information.
    fill_value
        Value to fill in for points outside of the convex hull of the input points.
    %(verbose)s

    Returns
    -------
    AnnData: A new AnnData object with the same metadata as the reference but with interpolated spatial data in `.X`.

    Examples
    --------
    Brings two spatial modalities measured on different coordinate grids (e.g.
    metabolomics and transcriptomics of the same slide) onto a shared set of
    locations. Here a coarser grid stands in for the reference:

    >>> import liana as li
    >>> target = li.ds.generate_toy_spatial()
    >>> reference = target[::2].copy()
    >>> interpolated = li.pp.interpolate_adata(target=target, reference=reference, spatial_key="spatial")
    """
    target_coords = target.obsm[spatial_key]
    reference_coords = reference.obsm[spatial_key]

    ad = AnnData(
        X=None,
        uns=dict(reference.uns),
        obs=get_obs(reference),
        var=get_var(target),
    )
    copy_aligned(ad, obsm=reference.obsm, obsp=reference.obsp, varm=target.varm)

    # `griddata` is stubbed for 1-D `values` but forwards `(npoints, ...)` to `LinearNDInterpolator`
    values: NDArray[np.floating] = _choose_mtx_rep(
        adata=target, use_raw=use_raw, layer=layer, verbose=verbose
    ).toarray()

    ad.X = csr_matrix(
        griddata(
            points=np.asarray(target_coords),
            xi=np.asarray(reference_coords),
            values=values,
            method=method,
            fill_value=fill_value,
        )
    )

    return ad
