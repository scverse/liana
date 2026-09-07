import pytest
from anndata import AnnData
from matplotlib.figure import Figure

from liana.plotting import annulus


def test_annulus(toy_spatial: AnnData) -> None:
    fig = annulus(
        toy_spatial,
        spatial_key="spatial",
        radius_step=200,
        annulus_steps=1,
        n_rings=5,
        seed=42,
    )

    assert isinstance(fig, Figure)
    assert annulus(toy_spatial, spatial_key="spatial", return_fig=False) is None

    with pytest.raises(KeyError, match="not found in adata.obsm"):
        annulus(toy_spatial, spatial_key="missing_key")
