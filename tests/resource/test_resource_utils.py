import decoupler as dc
import numpy as np
import pandas as pd
import pytest

from liana.resource import select_resource
from liana.resource._resource_utils import generate_lr_geneset


@pytest.mark.network
def test_generate_lr_resource() -> None:
    """Test generate_lr_resource."""
    # load data
    net = dc.op.progeny(top=1000, organism="human", thr_padj=1)
    resource = select_resource("consensus")
    lr_net = generate_lr_geneset(resource, net)
    assert set(lr_net.columns) == {"interaction", "weight", "source"}
    assert lr_net.shape[0] == 170
    assert lr_net["interaction"].nunique() == 153
    assert lr_net["source"].nunique() == 14
    assert np.isclose(lr_net[lr_net["interaction"] == "LAMB3^ITGAV_ITGB8"]["weight"].to_numpy()[0], 3.62299, atol=1e-5)


@pytest.mark.network
def test_generate_nondefault_lr_resource() -> None:
    """Test generate_lr_resource."""
    # load data
    net = dc.op.progeny(top=1000, organism="human")
    net.drop(columns=["weight"], inplace=True)
    net.rename(columns={"source": "tf", "target": "genesymbol"}, inplace=True)

    resource = select_resource("consensus")

    lr_net = generate_lr_geneset(resource, net, source="tf", weight=None, target="genesymbol")
    assert lr_net.shape[0] == 250
    assert "weight" not in lr_net.columns


def test_generate_lr_geneset_deduplicates_net() -> None:
    """A duplicated `net` row used to inflate the subunit count and drop the interaction entirely."""
    resource = pd.DataFrame({"ligand": ["LGALS9"], "receptor": ["PTPRC"]})
    net = pd.DataFrame(
        {
            "source": ["pathA", "pathA", "pathA"],
            "target": ["LGALS9", "PTPRC", "PTPRC"],
            "weight": [1.0, 1.0, 1.0],
        }
    )

    lr_net = generate_lr_geneset(resource, net)

    assert lr_net["interaction"].tolist() == ["LGALS9^PTPRC"]
    assert lr_net["weight"].tolist() == [1.0]

    # a duplicate that disagrees on the weight is ambiguous, not redundant
    net.loc[net.index[-1], "weight"] = -1.0
    with pytest.raises(ValueError, match="disagree on 'weight'"):
        generate_lr_geneset(resource, net)
