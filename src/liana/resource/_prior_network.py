from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from liana._core._common import _check_if_installed, _logg
from liana._core._constants import DefaultValues as V

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from corneto import Graph


def build_prior_network(
    ppis: pd.DataFrame | Sequence[tuple[str, float, str]],
    input_nodes: Mapping[str, float],
    output_nodes: Mapping[str, float],
    lr_sep: str | None = None,
    verbose: bool = V.verbose,
) -> Graph:
    """
    Build Prior Network from PPIs and input/output nodes.

    Parameters
    ----------
    ppis
        The PPIs to use for the prior network. If a list, each element must be a
        ``(source, mor, target)`` tuple where ``mor`` is +1 or -1. If a pandas
        DataFrame is provided, it must have the columns ``source``, ``mor``, and
        ``target``.
    input_nodes
        A dictionary of input nodes. The keys are the node names, the values are the node scores.
    output_nodes
        A dictionary of output nodes. The keys are the node names, the values are the node scores.
    lr_sep
        The separator to use to split the input nodes into ligand and receptor. If None, the input nodes will be used as is.
    verbose
        Whether to print progress information. Default: True

    Returns
    -------
    corneto.Graph

    Examples
    --------
    `ppis` is a signed protein-protein interaction network -- normally from
    OmniPath -- given either as `(source, sign, target)` tuples or as a `DataFrame`
    with `source`/`mor`/`target` columns. A tiny one is written out here so the
    example stays offline. `lr_sep` strips the ligand from an interaction name, so
    that `'HLA-DRA^CD4'` matches the receptor `'CD4'`:

    >>> import liana as li
    >>> ppis = [("CD4", 1, "LCK"), ("LCK", 1, "JUN"), ("LCK", -1, "FOS")]
    >>> prior = li.rs.build_prior_network(
    ...     ppis, input_nodes={"HLA-DRA^CD4": 1.0}, output_nodes={"JUN": 1.0, "FOS": -1.0}, lr_sep="^"
    ... )

    The network is pruned to what lies on a path from an input to an output. Hand
    the result to :func:`liana.mt.find_causalnet`.
    """
    _check_if_installed("corneto")  # raises a helpful ImportError if the extra is missing
    from corneto import Graph

    if lr_sep is not None:
        if any(lr_sep in k for k in input_nodes.keys()):
            _logg(
                f"Input nodes are in the format Ligand{lr_sep}Receptor. Extracting only the receptor...",
                verbose=verbose,
            )
            # Print only at most the first 3 entries
            for k, v in list(input_nodes.items())[:3]:
                _logg(f" - {k} -> {v}", verbose=verbose)
            _logg(" - ...", verbose=verbose)
            # Do the split only if lr_sep is present, otherwise get the same key:
            input_nodes = {k.split(lr_sep)[-1]: v for k, v in input_nodes.items()}

    _logg("Importing network...", verbose=verbose)
    if isinstance(ppis, pd.DataFrame):
        ppis = [(r.source, r.mor, r.target) for (_, r) in ppis.iterrows() if r.source != r.target]
    # corneto is unannotated, so name the type once at the boundary
    G: Graph = Graph.from_sif_tuples(ppis)
    _logg("done.", verbose=verbose)

    incl_inputs = set(G.vertices).intersection(set(input_nodes))
    incl_outputs = set(G.vertices).intersection(set(output_nodes))

    _logg(f" - Nodes x Edges: {G.num_vertices, G.num_edges}", verbose=verbose)
    _logg(f" - Provided inputs included in the prior network: {len(incl_inputs)}/{len(input_nodes)}", verbose=verbose)
    _logg(f" - Provided outputs included in the network: {len(incl_outputs)}/{len(output_nodes)}", verbose=verbose)

    _logg("Performing reachability analysis...", verbose=verbose)
    Gp: Graph = G.prune(list(incl_inputs), list(incl_outputs))
    _logg("done.", verbose=verbose)
    pruned_size = (Gp.num_vertices, Gp.num_edges)
    incl_inputs_pruned = set(Gp.vertices).intersection(incl_inputs)
    incl_outputs_pruned = set(Gp.vertices).intersection(incl_outputs)

    _logg(f" - Selected inputs: {len(incl_inputs_pruned)}/{len(incl_inputs)}.", verbose=verbose)
    _logg(f" - Selected outputs: {len(incl_outputs_pruned)}/{len(incl_outputs)}.", verbose=verbose)
    _logg(f" - Final size of the prior graph: {pruned_size}.", verbose=verbose)

    if len(incl_outputs_pruned) == 0:
        raise ValueError("None of the output nodes can be reached from the provided input nodes in the PPI network.")
    if len(incl_inputs_pruned) == 0:
        raise ValueError("None of the input nodes can reach any of the output nodes in the PPI network.")

    return Gp
