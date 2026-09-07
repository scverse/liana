from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from liana._core._constants import DefaultValues as V
from liana._core._constants import PrimaryColumns as P
from liana._core._docs import d


@d.dedent
def generate_lr_geneset(
    resource: DataFrame,
    net: DataFrame,
    ligand_key: str = P.ligand,
    receptor_key: str = P.receptor,
    lr_sep: str = V.lr_sep,
    source: str = "source",
    target: str = "target",
    weight: str | None = "weight",
) -> DataFrame:
    """
    Generate a ligand-receptor gene set from a resource and a network.

    Specifically, it works with weighted bipartite networks, where the weight represents the importance of the genes
    to a given geneset. The function will assign a weight to each ligand-receptor interaction, based on the mean.
    It does so by first assigning a weight to each ligand-receptor subunit, checking for sign coherence and completeness
    of the ligand-receptor complex.

    Parameters
    ----------
    resource:
        A pandas dataframe with [`ligand`, `receptor`] columns.
    net
        Prior knowledge network in bipartite or decoupler format.
    ligand
        Name of the ligand column in the resource
    receptor
        Name of the receptor column in the resource
    %(lr_sep)s
    source
        Name of the source column in the network.
    weight
        Name of the weight column in the network. If None, all weights are set to 1.

    Returns
    -------
    Returns ligand-receptor geneset resource as a pandas.DataFrame with the following columns:
    - interaction: ligand-receptor interaction
    - weight: mean weight of the interaction
    - source: source of the interaction

    Examples
    --------
    `net` is a bipartite gene set (e.g. pathways, transcription-factor regulons)
    in decoupler format. Only ligand-receptor pairs whose *both* partners are in
    the same gene set, with coherent signs, are kept:

    >>> import pandas as pd
    >>> import liana as li
    >>> resource = li.rs.select_resource("consensus")
    >>> net = pd.DataFrame(
    ...     {
    ...         "source": ["pathA", "pathA", "pathB", "pathB"],
    ...         "target": ["LGALS9", "PTPRC", "THY1", "ITGB2"],
    ...         "weight": [1.0, 1.0, -1.0, -1.0],
    ...     }
    ... )
    >>> geneset = li.rs.generate_lr_geneset(resource, net)
    >>> geneset
      source   interaction  weight
    0  pathA  LGALS9^PTPRC     1.0
    1  pathB    ITGB2^THY1    -1.0

    The result can then be handed to an enrichment method
    (e.g. `decoupler`) with the interaction names as features.
    """
    # TODO: Fix this if else, it's not very elegant
    if weight is None:
        weight = "weight"
        net[weight] = 1

        drop_weight = True
    else:
        drop_weight = False

    # supp keys
    ligand_weight = ligand_key + "_" + weight
    receptor_weight = receptor_key + "_" + weight
    ligand_source = ligand_key + "_" + source
    receptor_source = receptor_key + "_" + source

    # assign weights to each entity
    ligand_weights = _assign_entity_weights(resource, net, source=source, target=target, entity_key=ligand_key)
    ligand_weights.rename(columns={weight: ligand_weight, source: ligand_source}, inplace=True)
    receptor_weights = _assign_entity_weights(resource, net, source=source, target=target, entity_key=receptor_key)
    receptor_weights.rename(columns={weight: receptor_weight, source: receptor_source}, inplace=True)

    # join weights to the the ligand-receptor resource
    resource = resource.merge(ligand_weights, on=ligand_key, how="inner")
    resource = resource.merge(receptor_weights, on=receptor_key, how="inner")

    # keep only coherent ligand and receptor sources
    resource = resource[resource[ligand_source] == resource[receptor_source]]
    # mean of sign-coherent ligand-receptor weights
    resource[weight] = [
        _sign_coherent_mean(np.array([lig, rec]))
        for lig, rec in zip(resource[ligand_weight], resource[receptor_weight], strict=True)
    ]

    # unite ligand-receptor columns
    resource = resource.assign(interaction=lambda x: x[ligand_key] + lr_sep + x[receptor_key])

    # keep only relevant columns
    resource = resource[[ligand_source, "interaction", weight]].rename(columns={ligand_source: source})

    # drop nan weights
    resource = resource.dropna()

    if drop_weight:
        resource.drop(columns=["weight"], inplace=True)

    return resource


def _assign_entity_weights(
    resource: DataFrame,
    net: DataFrame,
    entity_key: str = "receptor",
    source: str = "source",
    target: str = "target",
    weight: str = "weight",
) -> DataFrame:
    # only keep relevant columns
    net = net[[source, target, weight]]

    # process ligand-receptor resource
    # assign receptor complex as entity
    entity_resource = resource[[entity_key]].drop_duplicates().set_index(entity_key)
    entity_resource["subunit"] = entity_resource.index
    # explode complexes, keeping the complex as a key
    entity_resource["subunit"] = entity_resource["subunit"].str.split("_")
    entity_resource = entity_resource.explode(["subunit"])

    # join weights to subunits
    entity_resource = entity_resource.reset_index()
    entity_resource = entity_resource.merge(net, left_on="subunit", right_on=target)

    # check for sign and set consistency
    # count expected subunits separated by _
    entity_resource = entity_resource.assign(subunit_expected=entity_resource[entity_key].str.count("_") + 1)
    # count subunits by receptor complex & source
    entity_resource["subunit_count"] = entity_resource.groupby([source, entity_key])[[weight]].transform("count")
    # check if all subunits are present
    entity_resource = entity_resource.assign(subunit_complete=lambda x: x["subunit_expected"] == x["subunit_count"])
    # assign flag to sign-coherent subunits
    entity_resource["sing_coherent"] = entity_resource.groupby([source, entity_key])[[weight]].transform(
        lambda x: np.all(x > 0) | np.all(x < 0)
    )

    # keep only relevant targets
    entity_resource = entity_resource[entity_resource["subunit_complete"]]  # keep only complete complexes
    entity_resource = entity_resource[entity_resource["sing_coherent"]]  # keep only sign-coherent complexes

    # get mean weight per complex & source
    entity_resource = entity_resource.groupby([source, entity_key])[[weight]].mean().reset_index()

    return entity_resource


def _sign_coherent_mean(x: NDArray[np.floating]) -> np.floating:
    """Mean of ``x``, or NaN when its entries disagree in sign."""
    if np.all(x > 0) | np.all(x < 0):
        return np.float64(np.mean(x))
    return np.float64(np.nan)
