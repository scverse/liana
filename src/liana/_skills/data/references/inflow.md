# Inflow: which cell types signal to which, at single-cell resolution

Aim: per receiver cell, a score for every (sender cell type, ligand, receptor) triple: bivariate
scores extended with the sender's identity. Needs cell-resolution data with cell-type labels (Xenium,
MERFISH, CosMx) or deconvolution proportions. Tutorials: `inflow_score`, `inflow_mofaflex`.

```python
li.pp.spatial_neighbors(adata, bandwidth=30)
lrdata = li.mt.inflow(adata, groupby="cell_type", resource_name="consensus", nz_prop=0.01)
li.mt.compute_global_specificity(lrdata, groupby="cell_type")   # groupby = receiver label
li.pl.dotplot(lrdata, uns_key="global_interactions", colour="lr_mean", size="pval", inverse_size=True)
li.pl.feature_by_group(lrdata, groupby="cell_type", labels=receivers, feature=lrdata.var_names[0])
```

- Bandwidth: one to two cell diameters (tutorials: 27 µm MERFISH, 60 µm for the programs analysis).
- Output is a **new** AnnData: cells x `source^ligand^receptor`, `.var` with `mean`, `variance`, `std`,
  `cv`, `nonzero_fraction`; `.obs`, `.obsm`, `.obsp` copied. Non-variable features are dropped.
- `nz_prop` is the main QC knob; sparse panels need low values (tutorials: 0.001); `nonzero_fraction`
  filters again afterwards.
- `global_interactions`: `source`, `ligand_complex`, `receptor_complex`, `target`, `lr_mean`, `pval`.
- `feature_by_group` needs `labels` (receivers) and one feature.

## Variants

- **Deconvolved spots**: `obsm_key="proportions"` instead of `groupby`; `adata.obsm[obsm_key]` is a
  DataFrame with cell types as columns (proportions or counts). Exactly one of `groupby`/`obsm_key`.
- **Transforms**: `x_transform`/`y_transform` (`li.pp.zi_minmax`) when ligand and receptor scales differ.
- **MuData**: `x_mod`/`y_mod` (ligand side, receptor side) with `x_use_raw`, `x_layer`.
- **Communication programs** (spatial gradients): filter `lrdata[:, lrdata.var["nonzero_fraction"] > 0.01]`,
  `li.ms.lrdata_to_mudata(lrdata, min_features=25, obs_keys=["cell_type"])` (one view per sender), fit
  MOFA-Flex (`mofaflex`, `torch`) with a Gaussian-process prior on the coordinates, map loadings back with
  `li.ms.get_variable_loadings(loadings=model.get_weights(), variable_sep="^",
  var_names=["source", "ligand_complex", "receptor_complex"])`. Scores are zero-inflated and
  heavy-tailed: widen the bandwidth or raise `nz_prop` if a few cells drive a factor. Factor signs are arbitrary.
- **Cheaper alternative**: single-cell methods with `spatial_key`, which weight cell-type pair scores
  by proximity instead of scoring each cell.
