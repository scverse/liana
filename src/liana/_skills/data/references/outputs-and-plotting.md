# liana_res, score semantics and plotting

## The result table

`adata.uns["liana_res"]` (or the returned DataFrame) has one row per source cell type, target cell
type and LR pair: `source`, `target`, `ligand_complex`, `receptor_complex`, `ligand`, `receptor`
(the least-expressed subunit), `ligand_props`, `receptor_props`, then the score columns.
`by_sample` adds a column named after `sample_key`.

| method | magnitude | specificity | lower is better |
|---|---|---|---|
| `rank_aggregate` | `magnitude_rank` | `specificity_rank` | both |
| `cellphonedb` | `lr_means` | `cellphone_pvals` | p-value |
| `cellchat` | `lr_probs` | `cellchat_pvals` | p-value |
| `geometric_mean` | `lr_gmeans` | `gmean_pvals` | p-value |
| `natmi` | `expr_prod` | `spec_weight` | neither |
| `connectome` | `expr_prod` | `scaled_weight` | neither |
| `logfc` | none | `lr_logfc` | neither |
| `singlecellsignalr` | `lrscore` | none | no |
| `scseqcomm` | `inter_score` | none | no |

`li.mt.get_method_scores()` returns this as `{score: lower_is_better}`;
`li.mt.process_scores(liana_res, score_key="magnitude_rank")` applies `-log10` to such scores so
that higher is better, which is what `lrs_to_views` and `to_tensor_c2c` do internally.

## Plots

All functions below except `circle_plot` and `feature_by_group` are plotnine and return a `ggplot`
(default `return_fig=True`): `p = li.pl.dotplot(...)`; `p.save("f.pdf")`; `p + p9.theme_bw()`.

```python
li.pl.dotplot(adata, colour="magnitude_rank", size="specificity_rank",
              inverse_colour=True, inverse_size=True,        # ranks/p-values: small is strong
              source_labels=["B", "CD4 T"], target_labels=["CD8 T"],
              top_n=20, orderby="magnitude_rank", orderby_ascending=True,
              filter_fn=lambda x: x["specificity_rank"] <= 0.05)
```

- `colour` and `size` are both required and must be columns of the result.
- `top_n` requires `orderby` and `orderby_ascending`.
- `filter_fn` keeps every row of an interaction if any of its rows passes, so a p-value filter
  still shows that interaction for all cell-type pairs.
- `source_labels` / `target_labels` raise if a label is absent.
- `li.pl.tileplot(adata, fill="means", label="props", ...)`: `fill` and `label` are suffixes that must
  exist as both `ligand_<x>` and `receptor_<x>` columns (e.g. `means`, `props`, or a `df_to_lr` stat).
- `li.pl.circle_plot(adata, groupby="cell_type", score_key="magnitude_rank", inverse_score=True,
  pivot_mode="counts"|"mean")` draws a cell-type network (matplotlib `Axes`).
- `li.pl.dotplot_by_sample(adata, sample_key="sample", colour=..., size=...)` facets interaction by
  sample; subset labels first or it explodes.
- Spatial results (`bivariate`, `inflow`) are AnnData objects, so use `sc.pl.spatial` /
  `sc.pl.embedding(basis="spatial", color="LIG^REC")` on them directly.
