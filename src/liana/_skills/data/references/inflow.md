# Inflow: which cell types signal to which, at single-cell resolution

Aim: per receiver cell, a score for every (sender cell type, ligand, receptor) triple, i.e. bivariate
interactions extended with the sender's identity. Needs cell-resolution data with cell-type labels
(Xenium, MERFISH, CosMx) or deconvolution proportions. Tutorials: `inflow_score`, `inflow_mofaflex`.
Unpublished extension (Alsayah et al., in preparation): cite LIANA+ meanwhile.

## Call

```python
li.pp.spatial_neighbors(adata, bandwidth=30)          # coordinate units, see below; cells keep set_diag=False
lrdata = li.mt.inflow(adata, groupby="cell_type", resource_name="mouseconsensus")
li.mt.compute_global_specificity(lrdata, groupby="cell_type")   # groupby = receiver label
lrdata.uns["global_interactions"]                     # source, target, ligand_complex, receptor_complex, lr_mean, pval
li.pl.dotplot(lrdata, uns_key="global_interactions", colour="lr_mean", size="pval", inverse_size=True)
li.pl.feature_by_group(lrdata, groupby="cell_type", feature="Astro^Apoe^Lrp1")   # one source^ligand^receptor
```

- Bandwidth is in coordinate units and should be one to two cell diameters (the tutorial uses 27 µm
  for MERFISH, 60 µm for the programs analysis); pick it with `li.pp.query_bandwidth`. Too wide blurs
  local patterns, too narrow misses gradients. Convert pixels to µm first (`spatial-bivariate.md`, graph section).
- Output is a **new** AnnData: cells x `"source^ligand^receptor"` features, with `.var` columns
  `mean`, `variance`, `std`, `cv`, `nonzero_fraction`; `.obs`, `.obsm`, `.obsp` copied from the input.
  Non-variable features are dropped.
- `nz_prop` (default 0.05) is the fraction of all cells where ligand and receptor are non-zero;
  sparse per-cell panels may need a lower value (the tutorials use 0.001);
  it is the main QC knob and `nonzero_fraction` lets you filter again afterwards.
- `compute_global_specificity` writes `lrdata.uns["global_interactions"]`; `groupby` is the receiver label.

## Variants

- **Deconvolved spots** (Visium with cell2location or similar): `obsm_key="proportions"` instead of
  `groupby`. `adata.obsm[obsm_key]` must be a `pandas.DataFrame` whose columns are the cell-type
  labels; values may be proportions or cell counts. Exactly one of `groupby` / `obsm_key`.
- **Transforms**: `x_transform` / `y_transform` are `None` by default and data dependent;
  `li.pp.zi_minmax` is the usual choice when ligand and receptor scales differ.
- **MuData**: `x_mod` / `y_mod` (ligand side, receptor side) plus `x_use_raw`, `x_layer`, same logic
  as in `single-cell-lr.md`.
- **Communication programs** (unsupervised, spatial gradients): filter
  `lrdata[:, lrdata.var["nonzero_fraction"] > 0.01]`, then `mdata = li.ms.lrdata_to_mudata(lrdata,
  min_features=25, obs_keys=["cell_type"])` (one view per sender), fit MOFA-Flex with a Gaussian
  process factor prior on the coordinates, and map loadings back with
  `li.ms.get_variable_loadings(loadings=model.get_weights(), variable_sep="^",
  var_names=["source", "ligand_complex", "receptor_complex"])`. Needs `mofaflex` and `torch`.
  Inflow scores are zero-inflated and heavy-tailed: widen the bandwidth or raise `nz_prop` if factors
  are driven by a handful of cells. A factor's sign is arbitrary.
- **Cheaper alternative**: the standard single-cell methods with `spatial_key` / `spatial_kwargs`
  (Variants in `single-cell-lr.md`), which weight cell-type pair scores by proximity instead of
  scoring each cell.
