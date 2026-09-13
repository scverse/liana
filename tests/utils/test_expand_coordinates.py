import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from tests._helpers import get_coordinates, get_obs

from liana.preprocessing import expand_coordinates


def create_test_adata(
    n_per_sample: int, n_samples: int, seed: int, spatial_key: str = "spatial", sample_key: str = "sample"
) -> AnnData:
    """
    Helper to create a test AnnData with several samples sharing overlapping coordinates.
    """
    rng = np.random.default_rng(seed)
    coords = np.concatenate([rng.uniform(0, 100, size=(n_per_sample, 2)) for _ in range(n_samples)])
    samples = np.repeat([f"sample_{i}" for i in range(n_samples)], n_per_sample)
    obs = pd.DataFrame({sample_key: pd.Categorical(samples)})
    adata = AnnData(X=np.zeros((n_per_sample * n_samples, 3)), obs=obs)
    adata.obsm[spatial_key] = coords
    return adata


@pytest.fixture
def adata() -> AnnData:
    return create_test_adata(n_per_sample=50, n_samples=3, seed=0)


def test_returns_new_object(adata: AnnData) -> None:
    expanded = expand_coordinates(adata, sample_key="sample", n_cols=2)
    assert isinstance(expanded, AnnData)
    assert expanded is not adata


def test_input_is_not_mutated(adata: AnnData) -> None:
    original = adata.obsm["spatial"].copy()
    expand_coordinates(adata, sample_key="sample", n_cols=2)
    np.testing.assert_array_equal(adata.obsm["spatial"], original)
    assert "spatial_original" not in adata.obsm


def test_original_coordinates_are_preserved(adata: AnnData) -> None:
    expanded = expand_coordinates(adata, sample_key="sample", n_cols=2)
    np.testing.assert_array_equal(expanded.obsm["spatial_original"], adata.obsm["spatial"])


def test_row_order_is_preserved(adata: AnnData) -> None:
    # each row must still map to its own sample after expansion
    expanded = expand_coordinates(adata, sample_key="sample", n_cols=2)
    assert expanded.obs_names.equals(adata.obs_names)
    # within a sample the expansion is a pure translation, so pairwise offsets are unchanged
    for sample in get_obs(adata)["sample"].cat.categories:
        mask = np.asarray(get_obs(adata)["sample"] == sample)
        before = get_coordinates(adata, "spatial")[mask]
        after = get_coordinates(expanded, "spatial")[mask]
        shift = after - before
        np.testing.assert_allclose(shift - shift[0], 0, atol=1e-9)


def test_grid_layout_prevents_overlap(adata: AnnData) -> None:
    # 3 samples, 2 columns -> [(row=0, col=0), (row=0, col=1), (row=1, col=0)]
    expanded = expand_coordinates(adata, sample_key="sample", n_cols=2)
    coords = expanded.obsm["spatial"]
    s = adata.obs["sample"].to_numpy()
    col0, col1, row1 = (coords[s == f"sample_{i}"] for i in range(3))

    assert col1[:, 0].min() > col0[:, 0].max()
    assert row1[:, 1].min() > col0[:, 1].max()


def test_single_sample_shifts_to_origin() -> None:
    adata = create_test_adata(n_per_sample=50, n_samples=1, seed=1)
    expanded = expand_coordinates(adata, sample_key="sample")
    np.testing.assert_allclose(get_coordinates(expanded, "spatial").min(axis=0), [0, 0], atol=1e-9)


def test_margin_increases_spacing(adata: AnnData) -> None:
    s = adata.obs["sample"].to_numpy()

    def gap(margin: float) -> float:
        expanded = expand_coordinates(adata, sample_key="sample", n_cols=2, margin=margin)
        coords = get_coordinates(expanded, "spatial")
        return float(coords[s == "sample_1"][:, 0].min() - coords[s == "sample_0"][:, 0].max())

    assert gap(1.0) > gap(0.0)


def test_custom_spatial_key(adata: AnnData) -> None:
    key = "custom_coords"
    adata.obsm[key] = adata.obsm.pop("spatial")
    expanded = expand_coordinates(adata, sample_key="sample", spatial_key=key, n_cols=2)
    assert f"{key}_original" in expanded.obsm
    assert "spatial" not in expanded.obsm


def test_raises_on_missing_sample_key(adata: AnnData) -> None:
    with pytest.raises(ValueError, match="not_a_column"):
        expand_coordinates(adata, sample_key="not_a_column")


def test_raises_on_missing_spatial_key(adata: AnnData) -> None:
    with pytest.raises(ValueError, match="not_a_key"):
        expand_coordinates(adata, sample_key="sample", spatial_key="not_a_key")


def test_unused_category_is_ignored(adata: AnnData) -> None:
    """A zero-observation level used to give an empty `sample_coords` and a zero-size reduction."""
    obs = get_obs(adata)
    obs["sample"] = obs["sample"].cat.add_categories(["sample_unused"])
    assert "sample_unused" in get_obs(adata)["sample"].cat.categories
    assert not (get_obs(adata)["sample"] == "sample_unused").any()

    baseline = expand_coordinates(create_test_adata(n_per_sample=50, n_samples=3, seed=0), sample_key="sample")
    expanded = expand_coordinates(adata, sample_key="sample")

    np.testing.assert_allclose(get_coordinates(expanded, "spatial"), get_coordinates(baseline, "spatial"))
