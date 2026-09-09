import pytest
from anndata import AnnData
from tests._helpers import plot_data as _frame

from liana.multisample import nmf
from liana.plotting import elbow


def test_elbow(toy_adata: AnnData) -> None:
    nmf(toy_adata, n_components=None, k_range=range(1, 5), random_state=0, max_iter=20)
    plot_data = _frame(elbow(toy_adata))
    assert plot_data["k"].to_list() == [1, 2, 3, 4]
    assert plot_data["error"].is_monotonic_decreasing


def test_elbow_raises_without_errors(toy_adata: AnnData) -> None:
    nmf(toy_adata, n_components=2, random_state=0, max_iter=20)
    with pytest.raises(ValueError, match="n_components=None"):
        elbow(toy_adata)
