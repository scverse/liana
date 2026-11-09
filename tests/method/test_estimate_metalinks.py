import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from tests._helpers import as_anndata, get_x

from liana.method import estimate_metalinks


def test_estimate_metalinks(toy_adata: AnnData) -> None:
    metabolites = ["A", "A", "B", "B"]
    resource = pd.DataFrame({"source": np.unique(metabolites), "receptor": ["TNFRSF4", "ITGB2"]})
    pd_net = pd.DataFrame(
        {"source": metabolites, "target": ["RBP7", "SRM", "MAD2L2", "AGTRAP"], "weight": [0.1, 0.2, 0.3, 1]}
    )
    t_net = pd.DataFrame(
        {"source": metabolites, "target": ["HES4", "TNFRSF4", "SSU72", "PARK7"], "weight": [0.1, 0.2, 0.3, 1]}
    )

    mdata = estimate_metalinks(toy_adata, resource, pd_net, t_net, use_raw=True, verbose=True, tmin=2)
    assert np.isin(["metabolite", "receptor"], list(mdata.mod.keys())).all()
    assert (mdata.var.index == ["A", "B", "ITGB2", "TNFRSF4"]).all()
    np.testing.assert_almost_equal(get_x(as_anndata(mdata.mod["metabolite"])).mean(), -0.18889697, decimal=5)
    np.testing.assert_almost_equal(get_x(as_anndata(mdata.mod["receptor"])).mean(), 0.7754228, decimal=5)


def test_estimate_metalinks_resource_columns(toy_adata: AnnData) -> None:
    """The metabolites can be in a `ligand` column, as from `get_metalinks(split=True)`, or the legacy `source` one."""
    pd_net = pd.DataFrame(
        {"source": ["A", "A", "B", "B"], "target": ["RBP7", "SRM", "MAD2L2", "AGTRAP"], "weight": 1.0}
    )
    resource = pd.DataFrame({"ligand": ["A", "B"], "receptor": ["TNFRSF4", "ITGB2"]})

    mdata = estimate_metalinks(toy_adata, resource, pd_net, use_raw=True, tmin=2)
    legacy = estimate_metalinks(toy_adata, resource.rename(columns={"ligand": "source"}), pd_net, use_raw=True, tmin=2)
    assert (mdata.var.index == ["A", "B", "ITGB2", "TNFRSF4"]).all()
    np.testing.assert_array_equal(
        get_x(as_anndata(mdata.mod["metabolite"])), get_x(as_anndata(legacy.mod["metabolite"]))
    )

    with pytest.raises(ValueError, match="'ligand'"):
        estimate_metalinks(toy_adata, resource.rename(columns={"ligand": "metabolite"}), pd_net, use_raw=True)
