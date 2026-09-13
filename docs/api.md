# API

Import liana as:

```python
import liana as li
```

The public API is organized into six namespaces: `li.mt` (methods), `li.pp`
(preprocessing), `li.ms` (multi-sample), `li.rs` (resources / prior knowledge),
`li.ds` (datasets), and `li.pl` (plotting).

## Methods (`li.mt`)

### Callable Ligand-Receptor Method instances

Ligand-receptor method instances provide helper functions and consistent
attributes, to describe each method instance, and are callable:

```{eval-rst}
.. module:: liana.mt
.. currentmodule:: liana.mt

.. autosummary::
    :toctree: generated

    cellchat.__call__
    cellphonedb.__call__
    connectome.__call__
    logfc.__call__
    natmi.__call__
    singlecellsignalr.__call__
    geometric_mean.__call__
    scseqcomm.__call__
    rank_aggregate.__call__
```

### Method metadata

```{eval-rst}
.. currentmodule:: liana.mt

.. autosummary::
    :toctree: generated

    show_methods
    get_method_scores
    process_scores
    AggregateClass
    aggregate_meta
```

### Local bivariate metrics

```{eval-rst}
.. currentmodule:: liana.mt

.. autosummary::
    :toctree: generated

    bivariate.__call__
    compute_global_specificity
```

### Spatial proximity & interaction metrics

```{eval-rst}
.. currentmodule:: liana.mt

.. autosummary::
    :toctree: generated

    cross_pcf.__call__
    lric.__call__
    inflow.__call__
    get_lric_auc
    get_lric_divergence
```

### Learn Spatial Relationships

```{eval-rst}
.. currentmodule:: liana.mt

.. autosummary::
    :toctree: generated

    MistyData
    MistyData.__call__
    genericMistyData
    lrMistyData
```

#### Single-view models

Passed as `model=` when calling a `MistyData` object:

```{eval-rst}
.. module:: liana.method.sp
.. currentmodule:: liana.method.sp

.. autosummary::
    :toctree: generated

    SingleViewModel
    LinearModel
    RandomForestModel
    RobustLinearModel
```

### Intracellular & multi-sample helpers

```{eval-rst}
.. currentmodule:: liana.mt

.. autosummary::
    :toctree: generated

    find_causalnet
    estimate_metalinks
    df_to_lr
```

## Preprocessing (`li.pp`)

```{eval-rst}
.. module:: liana.pp
.. currentmodule:: liana.pp

.. autosummary::
    :toctree: generated

    obsm_to_adata
    zi_minmax
    neg_to_zero
    spatial_neighbors
    spatial_pair_proximity
    expand_coordinates
    query_bandwidth
    interpolate_adata
```

## Multi-Sample (`li.ms`)

```{eval-rst}
.. module:: liana.ms
.. currentmodule:: liana.ms

.. autosummary::
    :toctree: generated

    to_tensor_c2c
    adata_to_views
    lrs_to_views
    lrdata_to_mudata
    filter_view_markers
    nmf
    estimate_elbow
    get_factor_scores
    get_variable_loadings
    mdata_to_anndata
```

## Prior knowledge (`li.rs`)

```{eval-rst}
.. module:: liana.rs
.. currentmodule:: liana.rs

.. autosummary::
    :toctree: generated

    select_resource
    show_resources
    generate_lr_geneset
    translate_resource
    translate_column
    get_hcop_orthologs
    get_metalinks
    describe_metalinks
    get_metalinks_values
    build_prior_network
```

## Datasets (`li.ds`)

```{eval-rst}
.. module:: liana.ds
.. currentmodule:: liana.ds

.. autosummary::
    :toctree: generated

    kang_2018
    kuppe_2022
    citeseq_pbmc5k
    vicari_2024
    yao_2023
    generate_toy_adata
    generate_toy_spatial
    generate_toy_mdata
    sample_lrs
```

## Visualization (`li.pl`)

```{eval-rst}
.. module:: liana.pl
.. currentmodule:: liana.pl

.. autosummary::
    :toctree: generated

    dotplot
    dotplot_by_sample
    tileplot
    elbow
    circle_plot
    connectivity
    misty_target_metrics
    misty_contributions
    misty_interactions
    annulus
    lric_lineplot
    lric_divergence
    feature_by_group
```
