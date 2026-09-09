# LRIC and cross-PCF: at what distance do interactions occur

Aim: for cell-type pairs (cross-PCF) or LR pairs (LRIC), an enrichment curve `g(r)` over distance:
`g > 1` co-enrichment, `≈ 1` random, `< 1` avoidance. Single-cell resolution data with coordinates.
Tutorial: `LRIC_tutorial`. Unpublished extension (Alsayah et al., in preparation): cite LIANA+ meanwhile.

## Three modes, two arguments

| mode | call | adds |
|---|---|---|
| tissue architecture only | `li.mt.cross_pcf(adata, groupby="cell_type")` | `source`, `target`, `radius`, `g` |
| LR, cell-type agnostic | `li.mt.lric(adata, resource_name=...)` | `ligand_complex`, `receptor_complex`, `interaction`, `radius`, `g` |
| LR, directed by cell type | `li.mt.lric(adata, resource_name=..., groupby="cell_type")` | plus `source`, `target`, `g_expr`, `g_pcf` |

In the directed mode `g_pcf` equals cross-PCF and `g` is the expression-weighted curve, so `g > 1` can
only come from cells that express the ligand and receptor. Results land in `adata.uns["lric"]` /
`adata.uns["cross_pcf"]` (`inplace=False` returns the DataFrame).

## Radii and filters

```python
li.pl.annulus_plot(adata)                                      # check the rings on the tissue first
li.mt.lric(adata, resource_name="mouseconsensus", groupby="cell_type")   # radii: max_radius, radius_step
```

- `max_radius`, `radius_step` (and `annulus_steps`) are in coordinate units; convert pixels to µm first
  (`spatial-bivariate.md`, graph section). The defaults assume µm
  and a cell diameter of about 20 µm; if coordinates are pixels, convert them or scale the radii.
  The `radius` column is the inner edge of each annulus; the first annulus starts at 0.
- `min_cells=None` drops cell types making up 1% or less of the cells; pass an integer to override.
- `nz_prop` applies in the agnostic mode (fraction of all cells), `expr_prop` in the directed mode
  (fraction within each cell type). Masked pairs stay as NaN rows.
- `groupby_pairs` and `cell_types` restrict the pairs; `pair_chunk` trades memory for speed.

## Summaries and plots

```python
li.mt.get_lric_auc(adata)            # per curve: score = mean log2 g over radii, peak_radius
li.pl.lric_lineplot(adata, interaction="Apoe^Lrp1", source="Astro", target="Neuron")
li.mt.get_lric_divergence(adata, feature_a={"source": "Astro", "target": "Neuron", "interaction": "Apoe^Lrp1"},
                          feature_b={"source": "Micro", "target": "Neuron", "interaction": "Apoe^Lrp1"})
li.pl.lric_divergence_plot(adata, feature_a=feature_a, feature_b=feature_b)
```

`get_lric_auc` output columns match `li.pl.dotplot`. Divergence compares two curves selected by
`{column: value}` dicts over any columns, so after concatenating results from several samples with a
`condition` column it compares the same interaction across conditions. `g` is floored at 0.05
before `log2`; pass `transform_fn=np.log2` to drop empty bins instead.
