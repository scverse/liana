from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from liana._core._common import _check_if_installed, _logg

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corneto import Graph
    from corneto.backend._base import ProblemDef


def _get_scores(d: Mapping[str, float]) -> tuple[list[float], list[float]]:
    return ([v for v in d.values() if v < 0], [v for v in d.values() if v > 0])


def find_causalnet(
    prior_graph: Graph,
    input_node_scores: Mapping[str, float],
    output_node_scores: Mapping[str, float],
    node_weights: Mapping[str, float] | None = None,
    node_cutoff: float = 0.1,
    min_penalty: float = 0.01,
    max_penalty: float = 1.0,
    missing_penalty: float = 10,
    edge_penalty: float = 0.01,
    solver: str | None = None,
    seed: int = 1337,
    max_runs: int = 1,
    stable_runs: int = 5,
    verbose: bool = True,
    **kwargs: object,
) -> tuple[pd.DataFrame | None, ProblemDef]:
    """
    Find the causal network that best explains the input/output node scores.

    Parameters
    ----------
    prior_graph : corneto.Graph
        The prior graph to use for the search.
    input_node_scores : dict
        A dictionary of input node scores.
    output_node_scores : dict
        A dictionary of output node scores.
    node_weights : dict, optional
        A dictionary of node weights. The keys are the node names, the values are the weights.
        If None, all nodes will have the same weight.
    node_cutoff : float
        The cutoff to use for the node weights. Nodes with a weight below this cutoff will be assigned
        the max_penalty, nodes with a weight above this cutoff will be assigned the min_penalty.
        Only used if node_weights is not None. Default: 0.1
    min_penalty : float
        The minimum penalty to assign to nodes with a weight above the cutoff.
        Only used if node_weights is not None. Default: 0.01
    max_penalty : float
        The maximum penalty to assign to nodes with a weight below the cutoff
        Only used if node_weights is not None. Default: 1.0
    missing_penalty : float
        The penalty to assign to nodes that are not measured. Default: 10
    edge_penalty : float
        The penalty to assign to edges. Default: 0.01
    solver : str, optional
        The solver to use. If None, the default solver will be used. Default: None
        It will default to the solver included in SCIPY, if no other solver is available.
    seed : int, optional
        The seed to use for the random number generator. Default: 1337
    max_runs : int, optional
        The maximum number of runs to perform. Consider increasing this value if the solver does not converge.
        In each run, the noise added to the edge and node penalties is perturbed slightly (iterating over the seed).
        By default, only 1 run is performed.
    stable_runs : int, optional
        The number of consecutive stable solutions requires to interrupt the iteration over max_runs. Only used if max_runs is not == 1. Default: 5
    verbose : bool, optional
        Whether to print progress information. Default: True
    **kwargs : dict, optional
        Additional arguments to pass to the solver.

    Returns
    -------
    df_all
        DataFrame containing the resulting causal network
    P
        Insantce of the Corneto problem definition

    Examples
    --------
    Takes the pruned graph from :func:`liana.rs.build_prior_network` and selects
    the sub-network whose signs are consistent with the input and output scores.
    This needs a mixed-integer solver; without one installed, `corneto` falls back
    to the solver bundled with SciPy:

    >>> import liana as li
    >>> ppis = [("CD4", 1, "LCK"), ("LCK", 1, "JUN"), ("LCK", -1, "FOS")]
    >>> prior = li.rs.build_prior_network(ppis, input_nodes={"CD4": 1.0}, output_nodes={"JUN": 1.0, "FOS": -1.0})
    >>> df, problem = li.mt.find_causalnet(
    ...     prior, input_node_scores={"CD4": 1.0}, output_node_scores={"JUN": 1.0, "FOS": -1.0}, verbose=False
    ... )
    """
    cn = _check_if_installed("corneto")

    if solver is None:
        solver = cn.methods.carnival.select_mip_solver()

    measured_nodes = set(input_node_scores.keys()) | set(output_node_scores.keys())

    _logg("Total positive/negative scores of the inputs and outputs:", verbose=verbose)
    w_neg_in, w_pos_in = _get_scores(input_node_scores)
    w_neg_out, w_pos_out = _get_scores(output_node_scores)
    _logg(f" - (-) input nodes: {sum(w_neg_in)}", verbose=verbose)
    _logg(f" - (+) input nodes: {sum(w_pos_in)}", verbose=verbose)
    _logg(f" - (-) output nodes: {sum(w_neg_out)}", verbose=verbose)
    _logg(f" - (+) output nodes: {sum(w_pos_out)}", verbose=verbose)

    # Total weights
    total = abs(sum(w_neg_in)) + abs(sum(w_neg_out)) + sum(w_pos_in) + sum(w_pos_out)
    _logg(f" - abs total (inputs + outputs): {total}", verbose=verbose)

    if node_weights is None:
        node_penalties: dict[str, float] = {}
    else:
        node_penalties = _weights_to_penalties(
            node_weights, cutoff=node_cutoff, max_penalty=max_penalty, min_penalty=min_penalty
        )

    run_count = 0  # total runs
    stable_count = 0  # stable solutions in a row
    df_all = None  # df with all solutions

    while run_count < max_runs:
        current_seed = seed + run_count
        if run_count > 0:
            _logg(f"Run {run_count} with seed {current_seed}", verbose=verbose)

        # assign 0 penalties to input/output nodes, missing_penalty to missing nodes
        # add a small amount of noise to the penalties to ensure reproducible solutions
        rng = np.random.default_rng(seed=current_seed)
        c_node_penalties = {
            k: node_penalties.get(k, missing_penalty) + rng.uniform(min_penalty / 20, min_penalty / 10)
            if k not in measured_nodes
            else 0.0
            for k in prior_graph.vertices
        }

        _logg("Building CORNETO problem...", verbose=verbose)
        P, G = cn.methods.carnival._extended_carnival_problem(
            prior_graph,
            input_node_scores,
            output_node_scores,
            node_penalties=c_node_penalties,
            edge_penalty=edge_penalty,
        )

        # E is the variable with 1 if edge activates or inhibits, 0 otherwise
        E = P.symbols["reaction_sends_activation_c0"] + P.symbols["reaction_sends_inhibition_c0"]
        W = rng.uniform(edge_penalty / 20, edge_penalty / 10, size=E.shape)
        P.add_objectives(W.T @ E)

        _logg(f"Solving with {solver}...", verbose=verbose)
        P.solve(solver=solver, verbosity=int(verbose), **kwargs)

        obj_names = ["Loss (unfitted inputs/output)", "Edge penalty error", "Node penalty error"]
        _logg("Solution summary:", verbose=verbose)
        for s, o in zip(obj_names, P.objectives, strict=False):
            _logg(f" - {s}: {o.value}", verbose=verbose)

        rows, cols = cn.methods.carnival.export_results(P, G, input_node_scores, output_node_scores)
        df = pd.DataFrame(rows, columns=cols)

        # Check if all rows from df are contained in df_all
        if df_all is None:
            df_all = df
            continue
        else:
            set_df = {tuple(row) for row in df.values}
            set_df_all = {tuple(row) for row in df_all.values}

            if set_df.issubset(set_df_all):
                stable_count += 1
            else:
                stable_count = 0

            df_all = pd.concat([df_all, df]).drop_duplicates()

        if stable_count >= stable_runs:
            break

        run_count += 1

    return df_all, P


def _weights_to_penalties(
    props: Mapping[str, float],
    cutoff: float,
    min_penalty: float,
    max_penalty: float,
) -> dict[str, float]:
    if any(p < 0 or p > 1 for p in props.values()):
        raise ValueError("Node weights were not between 0 and 1. Consider minmax or another normalization.")

    return {k: max_penalty if v < cutoff else min_penalty for k, v in props.items()}
