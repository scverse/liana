import pytest
from tests._helpers import as_frame

from liana.method import find_causalnet
from liana.resource import build_prior_network

input_pkn = [
    ("I1", 1, "N1"),
    ("N1", 1, "M1"),
    ("N1", 1, "M2"),
    ("I2", -1, "N2"),
    ("N2", -1, "M2"),
    ("N2", -1, "M1"),
]

input_scores = {"I1": 1, "I2": 1}
output_scores = {"M1": 1, "M2": 1}
node_weights = {"N1": 1, "N2": 1}


def test_build_prior_network() -> None:
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=True)
    assert prior_graph.num_vertices == 6
    assert prior_graph.num_edges == 6


def test_caulsalnet() -> None:
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=False)
    result, problem = find_causalnet(
        prior_graph,
        input_scores,
        output_scores,
        node_weights=node_weights,
        verbose=False,
        solver="scipy",
        seed=1337,
        max_runs=50,
        stable_runs=20,
    )

    assert problem.weights == [1.0, 0.01, 1.0, 1.0]
    df_res = as_frame(result)
    assert (df_res[df_res["target_type"] == "output"]["target"].isin(["M1", "M2"])).all()


def test_causalnet_noweights() -> None:
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=False)
    result, problem = find_causalnet(
        prior_graph,
        input_scores,
        output_scores,
        node_weights={"N1": 1, "N2": 0},
        verbose=False,
        solver="scipy",
        max_runs=50,
        stable_runs=20,
    )
    df_res = as_frame(result)
    assert df_res["source_pred_val"].to_numpy().sum() == 8
    assert df_res["target_pred_val"].to_numpy().sum() == 9


def test_causalnet_unknown_node_is_dropped_with_warning() -> None:
    """`build_prior_network` prunes nodes by design, so a scored leftover warns instead of raising."""
    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=False)
    with pytest.warns(UserWarning, match="absent from"):
        result, _ = find_causalnet(
            prior_graph,
            {**input_scores, "GHOST": 1},
            output_scores,
            verbose=False,
            solver="scipy",
        )

    df_res = as_frame(result)
    assert "GHOST" not in set(df_res["source"]) | set(df_res["target"])


def test_causalnet_failed_solve_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed solve used to surface as `None - None` inside corneto's `export_results`."""
    from types import SimpleNamespace

    from corneto.backend._base import ProblemDef

    def _infeasible(self: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status="infeasible")

    monkeypatch.setattr(ProblemDef, "solve", _infeasible)

    prior_graph = build_prior_network(input_pkn, input_scores, output_scores, verbose=False)
    with pytest.raises(RuntimeError, match="did not solve the problem"):
        find_causalnet(prior_graph, input_scores, output_scores, verbose=False, solver="scipy")
