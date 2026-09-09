"""Type aliases and narrowing helpers shared across liana.

`anndata` types :attr:`~anndata.AnnData.X` as `_XDataType | None` and :attr:`~anndata.AnnData.obs` as `DataFrame | Dataset2D`, because it also supports on-disk and lazily-backed objects.
liana only ever operates on in-memory arrays, so the helpers here narrow those unions once, at the point the data enters the package, and raise a clear error otherwise -- instead of every call site pushing the union around or silencing it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from fast_array_utils.types import CSBase
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData
    from mudata import MuData


type MatrixLike = NDArray[np.number] | CSBase
"""An expression matrix, dense or sparse -- the only shapes liana operates on."""

type ObsmValue = pd.DataFrame | NDArray[np.generic] | CSBase
"""A value :attr:`~anndata.AnnData.obsm` can hold.

anndata\'s public `AxisStorable` is wider than this (it also covers `uns`, so it
admits scalars, lists and dicts), while the `obsm` setter itself accepts only
frames and arrays.
"""


def _to_matrix(x: object, *, what: str) -> MatrixLike:
    """Narrow ``x`` to a :data:`MatrixLike`, or explain why it is not one."""
    if x is None:
        raise ValueError(f"`{what}` is empty; liana needs an expression matrix.")
    if not isinstance(x, np.ndarray | CSBase):
        raise TypeError(
            f"`{what}` must be an in-memory dense or sparse matrix, got {type(x).__name__}. "
            "Backed and lazily-loaded matrices are not supported; load it into memory first."
        )
    return x


def get_x(adata: AnnData) -> MatrixLike:
    """Return :attr:`~anndata.AnnData.X`, narrowed to an in-memory matrix."""
    return _to_matrix(adata.X, what="adata.X")


type RowFilter = Callable[[pd.Series], bool]
"""Predicate applied row-wise to a results frame (``df.apply(fn, axis=1)``)."""

type Aggregator = Callable[[pd.Series], object] | str
"""Reduces a group of values to one, as passed to ``DataFrame.agg``.

Either a callable (e.g. :func:`numpy.mean`) or the name of one (e.g. ``"mean"``).
"""

type SortKey = Callable[[pd.Series], pd.Series]
"""Transforms a column before sorting, as passed to ``DataFrame.sort_values(key=)``."""

type ScoreTransform = Callable[[pd.Series], pd.Series]
"""Rewrites a score column, e.g. to invert a "lower is stronger" score."""


def copy_aligned(
    target: AnnData,
    *,
    obsm: Mapping[str, ObsmValue] | None = None,
    obsp: Mapping[str, ObsmValue] | None = None,
    varm: Mapping[str, ObsmValue] | None = None,
) -> None:
    """Copy ``obsm``/``obsp``/``varm`` entries onto an already-built ``target``.

    :meth:`anndata.AnnData.__init__` types these arguments as ``Mapping[str, Sequence[Any]]``, which is narrower than what anndata actually stores there (frames and sparse matrices are neither).
    The per-key setters are typed correctly, so the entries are assigned one at a time instead.
    """
    for key, value in (obsm or {}).items():
        target.obsm[key] = value
    for key, value in (obsp or {}).items():
        target.obsp[key] = value
    for key, value in (varm or {}).items():
        target.varm[key] = value


def get_raw_x(adata: AnnData) -> MatrixLike:
    """Return ``adata.raw.X``, narrowed to an in-memory matrix."""
    if adata.raw is None:
        raise ValueError("`adata.raw` is not initialized.")
    return _to_matrix(adata.raw.X, what="adata.raw.X")


def get_coordinates(adata: AnnData | MuData, spatial_key: str) -> NDArray[np.float64]:
    """Return ``adata.obsm[spatial_key]`` as an ``(n_obs, n_dim)`` float array.

    `obsm` can hold frames and sparse matrices as well as arrays; every spatial function here needs plain dense coordinates, so the conversion and the accompanying shape check happen once, here.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"`adata.obsm['{spatial_key}']` not found; is the data spatial?")
    entry = adata.obsm[spatial_key]
    if isinstance(entry, CSBase):
        raise TypeError(f"`adata.obsm['{spatial_key}']` must be dense coordinates, got {type(entry).__name__}.")
    coordinates = np.asarray(entry, dtype=np.float64)
    if coordinates.ndim != 2:
        raise ValueError(f"`adata.obsm['{spatial_key}']` must be 2-dimensional, got {coordinates.ndim} dimension(s).")
    return coordinates


def get_obsm_frame(adata: AnnData, key: str) -> pd.DataFrame:
    """Return ``adata.obsm[key]``, narrowed to a :class:`~pandas.DataFrame`."""
    if key not in adata.obsm:
        raise KeyError(f"`adata.obsm['{key}']` not found.")
    entry = adata.obsm[key]
    if not isinstance(entry, pd.DataFrame):
        raise TypeError(f"`adata.obsm['{key}']` must be a DataFrame, got {type(entry).__name__}.")
    return entry


def _annotation(data: AnnData | MuData, axis: Literal["obs", "var"]) -> pd.DataFrame:
    frame = getattr(data, axis)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"`.{axis}` must be a pandas DataFrame, got {type(frame).__name__}. "
            "Lazily-backed (xarray) annotations are not supported."
        )
    return frame


def get_obs(data: AnnData | MuData) -> pd.DataFrame:
    """Return ``.obs``, narrowed to a :class:`~pandas.DataFrame`."""
    return _annotation(data, "obs")


def get_var(data: AnnData | MuData) -> pd.DataFrame:
    """Return ``.var``, narrowed to a :class:`~pandas.DataFrame`."""
    return _annotation(data, "var")
