from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pandas import DataFrame

from liana._core._constants import Keys as K

if TYPE_CHECKING:
    from types import ModuleType

    from anndata import AnnData

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _logg(message: str, level: str | None = "info", verbose: bool = False) -> None:
    """
    Log a message with a specified logging level.

    Parameters
    ----------
    message
        The message to log.
    level
        The logging level for the message (default is 'info'). Accepted levels
        are 'warn' or 'info', any other value will result in no logging.
    verbose
        Controls whether the message is logged or not.
    """
    if verbose:
        if level == "warn":
            logging.warning(message)
        elif level == "info":
            logging.info(message)


def _check_if_installed(package_name: str, custom_error_message: str | None = None) -> ModuleType:
    """
    Checks whether a package is installed in the current environment.

    Parameters
    ----------
    package_name
        The name of the package.
    custom_error_message
        Custom error message to display in case the package could not be found.

    Returns
    -------
    The imported module.

    Raises
    ------
    ImportError
        If the package could not be found/imported.
    """
    try:
        imported_module = __import__(package_name)
        return imported_module
    except ImportError:
        if custom_error_message:
            raise ImportError(custom_error_message) from None
        else:
            raise ImportError(
                f"{package_name} is not installed. Please install it with: \
                    pip install {package_name}"
            ) from None


def _get_liana_res(
    adata: AnnData | None,
    liana_res: DataFrame | None,
    uns_key: str = K.uns_key,
) -> DataFrame:
    if adata is not None:
        if uns_key not in adata.uns:
            raise KeyError(f"`{uns_key}` not found in `adata.uns`.")
        _logg(f"Using `adata.uns['{uns_key}']`")
        res = adata.uns[uns_key]
        if not isinstance(res, DataFrame):
            raise TypeError(f"`adata.uns['{uns_key}']` must be a DataFrame, got {type(res).__name__}.")
        return res.copy()
    if liana_res is not None:
        return liana_res.copy()
    raise ValueError("`liana_res` or AnnData with `uns_key` must be provided!")
