from itertools import product

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from fast_array_utils.conv import to_dense
from mudata import MuData
from scipy.sparse import csr_matrix
from tests._helpers import as_anndata, get_obs, get_x

from liana.datasets._sample_resource import sample_resource
from liana.method import inflow
from liana.preprocessing.transform import zi_minmax


def test_inflow_basic_structure(toy_spatial: AnnData) -> None:
    """Test basic execution and output structure preservation."""
    lrdata = as_anndata(inflow(toy_spatial, groupby="bulk_labels", resource_name="consensus", use_raw=True))

    # Check output structure
    assert isinstance(lrdata, type(toy_spatial))
    assert lrdata.shape == (toy_spatial.shape[0], 236)  # Fixed expected shape

    # Check var index format: "celltype^ligand^receptor"
    assert all("^" in idx for idx in lrdata.var_names)
    for idx in lrdata.var_names:
        parts = idx.split("^")
        assert len(parts) == 3
        celltype, ligand, receptor = parts
        assert celltype in get_obs(toy_spatial)["bulk_labels"].unique()

    # Check sparse matrix
    assert isinstance(get_x(lrdata), csr_matrix)

    # Check obs and obsm preserved
    assert get_obs(lrdata).equals(get_obs(toy_spatial))
    assert "spatial" in lrdata.obsm
    np.testing.assert_array_equal(lrdata.obsm["spatial"], toy_spatial.obsm["spatial"])

    # Check obsp preserved
    assert "spatial_connectivities" in lrdata.obsp


def test_inflow_with_transform(toy_spatial: AnnData) -> None:
    """Test inflow with zi_minmax transformation."""
    lrdata = as_anndata(
        inflow(
            toy_spatial,
            groupby="bulk_labels",
            resource_name="consensus",
            x_transform=zi_minmax,
            y_transform=zi_minmax,
            use_raw=False,
        )
    )

    assert lrdata.shape[0] == toy_spatial.shape[0]
    # Transformed values should be in [0, 1]
    assert get_x(lrdata).min() >= 0
    assert get_x(lrdata).max() <= 1


def test_inflow_nz_prop_filter(toy_spatial: AnnData) -> None:
    """Test filtering by non-zero proportion."""
    # Strict filter
    lrdata_strict = as_anndata(
        inflow(toy_spatial, groupby="bulk_labels", resource_name="consensus", nz_prop=0.2, use_raw=True)
    )

    # Lenient filter
    lrdata_lenient = as_anndata(
        inflow(toy_spatial, groupby="bulk_labels", resource_name="consensus", nz_prop=0.001, use_raw=True)
    )

    # Strict filter should have fewer or equal interactions
    assert lrdata_strict.shape[1] <= lrdata_lenient.shape[1]


def test_inflow_custom_resource(toy_spatial: AnnData) -> None:
    """Test with custom resource DataFrame."""
    resource = sample_resource(toy_spatial, n_lrs=10)

    lrdata = as_anndata(inflow(toy_spatial, groupby="bulk_labels", resource=resource, use_raw=True))

    assert lrdata.shape[1] > 0
    assert lrdata.shape[0] == toy_spatial.shape[0]


def test_inflow_numerical_correctness(toy_spatial: AnnData) -> None:
    """Test numerical correctness of inflow scores."""
    lrdata = as_anndata(inflow(toy_spatial, groupby="bulk_labels", resource_name="consensus", use_raw=True))

    # Check specific numerical values (regression test)
    np.testing.assert_almost_equal(get_x(lrdata).mean(), 0.053451, decimal=3)
    np.testing.assert_almost_equal(get_x(lrdata).sum(), 8830.041322, decimal=3)


def test_inflow_missing_connectivity(toy_spatial: AnnData) -> None:
    """Test error when spatial_connectivities is missing."""
    # Remove spatial connectivity
    del toy_spatial.obsp["spatial_connectivities"]

    with pytest.raises(ValueError, match="spatial_connectivities"):
        inflow(toy_spatial, groupby="bulk_labels", resource_name="consensus", use_raw=True)


def test_inflow_no_features_pass_filter(toy_spatial: AnnData) -> None:
    """Test error when no features pass nz_prop filter."""
    with pytest.raises(ValueError, match="No features passed"):
        inflow(
            toy_spatial,
            groupby="bulk_labels",
            resource_name="consensus",
            nz_prop=0.99,  # Very strict filter
            use_raw=True,
        )


def test_inflow_invalid_groupby(toy_spatial: AnnData) -> None:
    """Test error with invalid groupby column."""
    with pytest.raises(KeyError):
        inflow(toy_spatial, groupby="nonexistent_column", resource_name="consensus", use_raw=True)


def test_inflow_with_obsm_key(toy_spatial: AnnData) -> None:
    """Test inflow with pre-computed cell type matrix from obsm."""

    # Create soft cell type assignments (probabilities) as DataFrame
    n_celltypes = 3
    ct_probs = np.random.rand(toy_spatial.n_obs, n_celltypes)
    ct_probs = ct_probs / ct_probs.sum(axis=1, keepdims=True)  # normalize to sum to 1
    ct_probs_df = pd.DataFrame(
        ct_probs, columns=[f"CT_{i}" for i in range(n_celltypes)], index=get_obs(toy_spatial).index
    )
    toy_spatial.obsm["ct_probs"] = ct_probs_df

    lrdata = as_anndata(inflow(toy_spatial, obsm_key="ct_probs", resource_name="consensus", use_raw=True))

    # Check output structure
    assert isinstance(lrdata, type(toy_spatial))
    assert lrdata.shape[0] == toy_spatial.shape[0]
    assert lrdata.shape[1] > 0


def test_inflow_obsm_vs_groupby_equivalence(toy_spatial: AnnData) -> None:
    """Test that one-hot from groupby matches binary obsm."""
    import pandas as pd

    # Create one-hot from groupby
    ct_onehot = pd.get_dummies(get_obs(toy_spatial)["bulk_labels"])
    toy_spatial.obsm["ct_onehot"] = ct_onehot

    lrdata1 = as_anndata(inflow(toy_spatial, groupby="bulk_labels", resource_name="consensus", use_raw=True))
    lrdata2 = as_anndata(inflow(toy_spatial, obsm_key="ct_onehot", resource_name="consensus", use_raw=True))

    # Should be identical (or very close)
    assert lrdata1.shape == lrdata2.shape
    np.testing.assert_array_almost_equal(to_dense(get_x(lrdata1)), to_dense(get_x(lrdata2)), decimal=5)


def test_inflow_groupby_obsm_validation(toy_spatial: AnnData) -> None:
    """Test error when neither or both groupby/obsm_key provided."""
    # Test neither parameter provided
    with pytest.raises(ValueError, match="Exactly one"):
        inflow(toy_spatial, resource_name="consensus", use_raw=True)

    # Test both parameters provided
    toy_spatial.obsm["ct"] = np.random.rand(toy_spatial.n_obs, 3)
    with pytest.raises(ValueError, match="Exactly one"):
        inflow(toy_spatial, groupby="bulk_labels", obsm_key="ct", resource_name="consensus", use_raw=True)


def test_inflow_obsm_missing_key(toy_spatial: AnnData) -> None:
    """Test error when obsm_key not found in obsm."""
    with pytest.raises(KeyError, match="not found in adata.obsm"):
        inflow(toy_spatial, obsm_key="nonexistent_key", resource_name="consensus", use_raw=True)


def test_inflow_obsm_not_dataframe(toy_spatial: AnnData) -> None:
    """Test error when obsm matrix is not a DataFrame."""
    # Create matrix as numpy array instead of DataFrame
    toy_spatial.obsm["ct_array"] = np.random.rand(toy_spatial.n_obs, 3)

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        inflow(toy_spatial, obsm_key="ct_array", resource_name="consensus", use_raw=True)


def test_inflow_with_mudata(toy_mdata: MuData) -> None:
    """Test inflow with MuData input."""
    interactions = list(product(toy_mdata.mod["adata_x"].var.index, toy_mdata.mod["adata_y"].var.index))

    lrdata = as_anndata(
        inflow(
            toy_mdata,
            groupby="bulk_labels",
            interactions=interactions,
            x_mod="adata_x",
            y_mod="adata_y",
            x_use_raw=False,
            y_use_raw=False,
            nz_prop=0,
        )
    )

    # Check output structure
    assert isinstance(lrdata, type(toy_mdata.mod["adata_x"]))
    assert lrdata.shape[0] == toy_mdata.shape[0]
    assert lrdata.shape[1] > 0

    # Check var index format: "celltype^ligand^receptor"
    assert all("^" in idx for idx in lrdata.var_names)

    # Check sparse matrix
    assert isinstance(get_x(lrdata), csr_matrix)


def test_inflow_mudata_vs_anndata_equivalence(toy_mdata: MuData) -> None:
    """Test that MuData and AnnData give same results when data is identical."""
    from liana.multisample.mdata_to_anndata import mdata_to_anndata

    interactions = list(product(toy_mdata.mod["adata_x"].var.index, toy_mdata.mod["adata_y"].var.index))

    # Run with MuData
    lrdata_mudata = as_anndata(
        inflow(
            toy_mdata,
            groupby="bulk_labels",
            interactions=interactions,
            x_mod="adata_x",
            y_mod="adata_y",
            x_use_raw=False,
            y_use_raw=False,
            nz_prop=0,
        )
    )

    # Convert to AnnData manually and run
    adata_combined = mdata_to_anndata(
        toy_mdata, x_mod="adata_x", y_mod="adata_y", x_use_raw=False, y_use_raw=False, verbose=False
    )

    lrdata_anndata = as_anndata(
        inflow(adata_combined, groupby="bulk_labels", interactions=interactions, use_raw=False, layer=None, nz_prop=0)
    )

    # Check that results have the same dimensions
    assert lrdata_mudata.shape == lrdata_anndata.shape

    # Check that variable names match
    assert set(lrdata_mudata.var_names) == set(lrdata_anndata.var_names)


def test_inflow_mudata_missing_mod(toy_mdata: MuData) -> None:
    """Test error handling when modality parameters are missing for MuData."""
    interactions = list(product(toy_mdata.mod["adata_x"].var.index, toy_mdata.mod["adata_y"].var.index))

    # Missing x_mod
    with pytest.raises(ValueError, match="requires 'x_mod' and 'y_mod'"):
        inflow(
            toy_mdata,
            groupby="bulk_labels",
            interactions=interactions,
            y_mod="adata_y",
            x_use_raw=False,
            y_use_raw=False,
        )

    # Missing y_mod
    with pytest.raises(ValueError, match="requires 'x_mod' and 'y_mod'"):
        inflow(
            toy_mdata,
            groupby="bulk_labels",
            interactions=interactions,
            x_mod="adata_x",
            x_use_raw=False,
            y_use_raw=False,
        )


def custom_transform_with_kwargs(mat: csr_matrix, clip_max: float = 1.0) -> csr_matrix:
    """Custom transform that uses kwargs."""
    from liana.preprocessing.transform import zi_minmax

    transformed = zi_minmax(mat)
    # Clip to a custom max value
    transformed.data = np.clip(transformed.data, 0, clip_max)
    return transformed


def test_anndata_transform_kwargs(toy_spatial: AnnData) -> None:
    """Test x_transform_kwargs and y_transform_kwargs with AnnData."""
    # Test with custom clip value - same for both x and y
    lrdata = as_anndata(
        inflow(
            toy_spatial,
            groupby="bulk_labels",
            resource_name="consensus",
            x_transform=custom_transform_with_kwargs,
            y_transform=custom_transform_with_kwargs,
            x_transform_kwargs={"clip_max": 0.5},
            y_transform_kwargs={"clip_max": 0.5},
            use_raw=False,
        )
    )

    # Verify clipping worked (should be <= 0.5 * 0.5 = 0.25 for product)
    assert get_x(lrdata).max() <= 0.26, f"Expected max <= 0.26, got {get_x(lrdata).max()}"


def test_mudata_transform_kwargs(toy_mdata: MuData) -> None:
    """Test x_transform_kwargs and y_transform_kwargs with MuData."""
    interactions = list(product(toy_mdata.mod["adata_x"].var.index, toy_mdata.mod["adata_y"].var.index))

    # Test with different clip values for x and y
    lrdata = as_anndata(
        inflow(
            toy_mdata,
            groupby="bulk_labels",
            interactions=interactions,
            x_mod="adata_x",
            y_mod="adata_y",
            x_transform=custom_transform_with_kwargs,
            y_transform=custom_transform_with_kwargs,
            x_transform_kwargs={"clip_max": 0.3},  # Custom parameter for x
            y_transform_kwargs={"clip_max": 0.7},  # Custom parameter for y
            x_use_raw=False,
            y_use_raw=False,
            nz_prop=0,
        )
    )
    assert get_x(lrdata).max() <= 0.22, f"Expected max <= 0.22, got {get_x(lrdata).max()}"
