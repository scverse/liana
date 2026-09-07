from __future__ import annotations

import pathlib

from numpy import unique
from pandas import DataFrame, RangeIndex, read_csv

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V


def select_resource(resource_name: str = V.resource_name) -> DataFrame:
    """
    Read resource of choice from the pre-generated resources in LIANA.

    Parameters
    ----------
    resource_name
        Name of the resource to be loaded and use for ligand-receptor inference.

    Raises
    ------
    ValueError
        If the resource name provided is not availabe in LIANA

    Returns
    -------
    A dataframe with ``['ligand', 'receptor']`` columns

    Examples
    --------
    The `'consensus'` resource ships with LIANA+ and is the one used by default:

    >>> import liana as li
    >>> resource = li.rs.select_resource("consensus")
    >>> resource.head(3)
       ligand receptor
    0  LGALS9    PTPRC
    1  LGALS9      MET
    2  LGALS9     CD44

    Pass the frame to any method via `resource=`.
    """
    resource_name = resource_name.lower()

    resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")

    resource = read_csv(resource_path, index_col=False)

    if resource_name not in resource["resource"].unique():
        raise ValueError(f"Resource {resource_name} not found. Please choose from {resource['resource'].unique()}")

    resource = resource[resource["resource"] == resource_name]

    resource = resource[["source_genesymbol", "target_genesymbol"]]
    resource = resource.rename(columns={"source_genesymbol": "ligand", "target_genesymbol": "receptor"})

    return resource


def show_resources() -> list[str]:
    """
    Show available resources.

    Returns
    -------
    A list of resource names available via ``liana.resource.select_resource``

    Examples
    --------
    Lists the resource names that :func:`liana.rs.select_resource` accepts --
    `'consensus'`, `'cellphonedb'`, `'cellchatdb'` and a dozen others:

    >>> import liana as li
    >>> resources = li.rs.show_resources()
    """
    resource_path = pathlib.Path(__file__).parent.joinpath("omni_resource.csv")
    resource = read_csv(resource_path, index_col=False)
    return list(unique(resource["resource"]))


def _handle_resource(
    interactions: list[tuple[str, str]] | None = None,
    resource: DataFrame | None = None,
    resource_name: str | None = None,
    x_name: str = "ligand",
    y_name: str = "receptor",
    verbose: bool = True,
) -> DataFrame:
    if interactions is None:
        if resource is None:
            if resource_name is None:
                raise ValueError("If 'interactions' and 'resource' are both None, 'resource_name' must be provided.")
            else:
                _logg(f"Using resource `{resource_name}`.", verbose=verbose)
                resource = select_resource(resource_name)
        else:
            if verbose:
                print("Using provided `resource`.")
            if x_name not in resource.columns or y_name not in resource.columns:
                raise ValueError(
                    "If 'interactions' is None, 'resource' must be a valid DataFrame "
                    f"with columns '{x_name}' and '{y_name}'."
                )
            resource = resource.copy()
            resource = resource.dropna(subset=[x_name, y_name]).drop_duplicates()
            resource.index = RangeIndex(len(resource))
            resource.index.name = None
    else:
        _logg("Using provided `interactions`.", verbose=verbose)
        if any(len(item) != 2 for item in interactions):
            raise ValueError("'interactions' should be a list of tuples in the format [(x1, y1), (x2, y2), ...].")
        resource = DataFrame(set(interactions), columns=[x_name, y_name])

    return resource
