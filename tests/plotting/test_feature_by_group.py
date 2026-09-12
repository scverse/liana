import numpy as np
import pytest
from anndata import AnnData
from tests._helpers import get_obs, not_none

from liana.plotting import feature_by_group


def test_feature_by_group(toy_spatial: AnnData) -> None:
    labels = ["Dendritic", "CD56+ NK"]
    _, ax = not_none(
        feature_by_group(
            adata=toy_spatial,
            groupby="bulk_labels",
            labels=labels,
            feature="HES4",
            normalize=True,
            percentile_scaling=(5, 95),
            show_counts=True,
        )
    )

    # a background layer of every spot, then one layer per requested label
    # holding exactly that label's cells, at their spatial coordinates
    assert len(ax.collections) == len(labels) + 1
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), toy_spatial.obsm["spatial"])

    for label, layer in zip(labels, ax.collections[1:], strict=True):
        mask = np.asarray(get_obs(toy_spatial)["bulk_labels"] == label)
        np.testing.assert_array_equal(layer.get_offsets(), toy_spatial.obsm["spatial"][mask])
        # normalize=True puts the colour values on a 0-1 scale
        values = not_none(layer.get_array())
        assert values.min() >= 0 and values.max() <= 1


def test_feature_by_group_no_normalize_autoscales(toy_spatial: AnnData) -> None:
    """`set_clim(0, 1)` used to apply to the unnormalised values too, saturating the plot."""
    labels = ["Dendritic", "CD56+ NK"]

    _, ax = not_none(
        feature_by_group(adata=toy_spatial, groupby="bulk_labels", labels=labels, feature="HES4", normalize=True)
    )
    assert [layer.get_clim() for layer in ax.collections[1:]] == [(0, 1)] * len(labels)

    _, ax = not_none(
        feature_by_group(adata=toy_spatial, groupby="bulk_labels", labels=labels, feature="HES4", normalize=False)
    )
    # the colorbar now spans what is drawn, which here exceeds the old fixed range
    scaled = [layer for layer in ax.collections[1:] if not_none(layer.get_array()).max() > 0]
    assert scaled, "no label carries expression to autoscale"
    for layer in scaled:
        values = not_none(layer.get_array())
        assert layer.get_clim() != (0, 1)
        assert layer.get_clim()[1] == pytest.approx(values.max())
        assert values.max() > 1


def test_feature_by_group_skips_empty_labels(toy_spatial: AnnData) -> None:
    # a label without cells is skipped rather than raising
    toy_spatial.obs["bulk_labels"] = toy_spatial.obs["bulk_labels"].cat.add_categories("Empty")
    feature_by_group(adata=toy_spatial, groupby="bulk_labels", labels=["Dendritic", "Empty"], feature="HES4")


def test_feature_by_group_raises(toy_spatial: AnnData) -> None:
    with pytest.raises(ValueError, match="'labels' must contain at least one label"):
        feature_by_group(adata=toy_spatial, groupby="bulk_labels", labels=[], feature="HES4")

    with pytest.raises(KeyError, match="`not_a_column` not found in `adata.obs.columns`"):
        feature_by_group(adata=toy_spatial, groupby="not_a_column", labels=["Dendritic"], feature="HES4")

    with pytest.raises(KeyError, match=r"`adata.obsm\['not_a_key'\]` not found"):
        feature_by_group(
            adata=toy_spatial, groupby="bulk_labels", labels=["Dendritic"], feature="HES4", spatial_key="not_a_key"
        )
