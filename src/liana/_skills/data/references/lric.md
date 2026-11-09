# LRIC and cross-PCF: at what distance do interactions occur

Aim: an enrichment curve `g(r)` over distance for cell-type pairs (cross-PCF) or LR pairs (LRIC):
`g > 1` co-enrichment, `≈ 1` random, `< 1` avoidance. Single-cell data with coordinates; AnnData only.
Tutorial: `LRIC_tutorial`.

| mode | call | adds to `uns` |
|---|---|---|
| tissue architecture only | `li.mt.cross_pcf(adata, groupby="cell_type")` | `cross_pcf`: `source`, `target`, `interaction`, `radius`, `g` |
| LR, cell-type agnostic | `li.mt.lric(adata, resource_name=...)` | `lric`: `ligand_complex`, `receptor_complex`, `interaction`, `radius`, `g` |
| LR, directed by cell type | `li.mt.lric(..., groupby="cell_type")` | plus `source`, `target`, `g_expr`, `g_pcf` |

In the directed mode `g_pcf` is the cross-PCF and `g` the expression-weighted curve, so `g > 1` comes
only from cells expressing ligand and receptor. `inplace=False` returns the DataFrame.

```python
li.pl.annulus(adata, radius_step=10)                         # check the rings on the tissue first
li.mt.lric(adata, groupby="cell_type", resource_name="consensus", max_radius=100, radius_step=10)
auc = li.mt.get_lric_auc(adata)                              # per curve: score (mean log2 g), peak_radius
top = auc.iloc[0]
li.pl.lric_lineplot(adata, interaction=top["interaction"], source=top["source"], target=top["target"])
feature_a = {"source": top["source"], "target": top["target"], "interaction": top["interaction"]}
senders = auc[(auc["interaction"] == top["interaction"]) & (auc["target"] == top["target"])]["source"]
feature_b = {**feature_a, "source": senders.iloc[1]}           # same LR and receiver, another sender
li.mt.get_lric_divergence(adata, feature_a=feature_a, feature_b=feature_b)
li.pl.lric_divergence(adata, feature_a=feature_a, feature_b=feature_b)
```

- `max_radius`, `radius_step` are in coordinate units (defaults assume µm, ~20 µm cells).
  `annulus_steps` is an integer ring width in multiples of `radius_step` (smooths sparse curves).
  `radius` is each annulus' inner edge; the first starts at 0.
- `min_cells=None` drops cell types making up ≤ 1% of cells; pass an integer to override.
- Masked pairs (filtered by `nz_prop` in the agnostic mode, `expr_prop` in the directed one) stay as NaN rows.
- `groupby_pairs`, `cell_types` restrict which pairs are reported, not which cells are used, so they do not speed it up.
- Per-gene rescaling cancels in `g(r)`; for a nonlinear weight (e.g. `sqrt`) pass a `layer=`.
- `get_lric_auc` output fits `li.pl.dotplot`; it raises on results stacked over samples (summarise one
  at a time). Divergence selects curves by `{column: value}`, so it can compare conditions after concatenating
  samples. `g` is floored at 0.05 before `log2`; `transform_fn=np.log2` drops empty bins instead.
