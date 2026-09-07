from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData

from liana._core._docs import d
from liana._core._types import ObsmValue, copy_aligned, get_obs

if TYPE_CHECKING:
    from collections.abc import Mapping


@d.dedent
def obsm_to_adata(
    adata: AnnData,
    obsm_key: str,
    df: pd.DataFrame | None = None,
    _uns: dict[str, pd.DataFrame] | None = None,
    _obsm: dict[str, pd.DataFrame] | None = None,
    _var: pd.DataFrame | None = None,
) -> AnnData:
    """
    Extracts a dataframe from adata.obsm and returns a new AnnData object with the values stored in X.

    Parameters
    ----------
    %(adata)s
    obsm_key
        `.osbm` key to extract.
    df
        Dataframe with stats per cell/spot. If None, it will be extracted from
        adata.obsm[obsm_key].
    _uns
        Dictionary with uns data. If None, it will be extracted from adata.uns.
    _obsm
        Dictionary with obsm data. If None, it will be extracted from
        adata.obsm.
    _var
        Dataframe with var data. If None, it will be extracted from adata.var.

    Returns
    -------
    An AnnData object with the values stored in X.

    Examples
    --------
    Turns an `.obsm` entry into an `AnnData` of its own, so that its columns can be
    treated as variables by functions that expect features in `.X`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> pca = li.pp.obsm_to_adata(adata, "X_pca")
    """
    if df is None:
        entry = adata.obsm[obsm_key]
        # Keep the frame's own columns when there are any -- they become `var_names`.
        df = pd.DataFrame(entry) if isinstance(entry, pd.DataFrame) else pd.DataFrame(np.asarray(entry))

    obs = get_obs(adata)
    uns = dict(adata.uns) if _uns is None else _uns
    obsm: Mapping[str, ObsmValue] = adata.obsm if _obsm is None else _obsm
    var = pd.DataFrame(index=df.columns) if _var is None else _var

    X = np.array(df, dtype=np.float32)

    adata_out = AnnData(X=X, obs=obs, var=var, uns=uns)
    copy_aligned(adata_out, obsm=obsm, obsp=adata.obsp)
    return adata_out
