# liana_res, score semantics and plotting

## The result table

One row per source, target and LR pair: `source`, `target`, `ligand_complex`, `receptor_complex`,
`ligand`, `receptor`, per-entity `ligand_means`/`_props`,
`receptor_means`/`_props`, then score columns. `rank_aggregate` adds every aggregated method's scores plus
`magnitude_rank`, `specificity_rank`; method-specific intermediates (`ligand_zscores`, `ligand_logfc`)
only via `supp_columns=`. `by_sample` adds the `sample_key` column. Missing pairs were filtered by `expr_prop`.

| method | magnitude | specificity | lower is better |
|---|---|---|---|
| `rank_aggregate` | `magnitude_rank` | `specificity_rank` | both |
| `cellphonedb` | `lr_means` | `cellphone_pvals` | p-value |
| `cellchat` | `lr_probs` | `cellchat_pvals` | p-value |
| `geometric_mean` | `lr_gmeans` | `gmean_pvals` | p-value |
| `natmi` | `expr_prod` | `spec_weight` | neither |
| `connectome` | `expr_prod` | `scaled_weight` | neither |
| `logfc` | none | `lr_logfc` | neither |
| `singlecellsignalr` | `lrscore` | none | neither |
| `scseqcomm` | `inter_score` | none | neither |

`li.mt.get_method_scores()` returns `{score: lower_is_better}`; `li.mt.process_scores(liana_res,
score_key=...)` applies `-log10` so higher is better (multi-sample helpers do this internally).

## Plots

```python
p = li.pl.dotplot(adata, colour="magnitude_rank", size="specificity_rank",
                  inverse_colour=True, inverse_size=True,          # small rank or p-value = strong
                  source_labels=sources, target_labels=targets,
                  top_n=20, orderby="magnitude_rank", orderby_ascending=True,
                  filter_fn=lambda row: row["specificity_rank"] <= 0.05)
p.save("dotplot.pdf")
li.pl.tileplot(adata, fill="means", label="props", top_n=10, orderby="magnitude_rank", orderby_ascending=True)
li.pl.circle(adata, groupby="cell_type", score_key="magnitude_rank", inverse_score=True)
```

- `dotplot`: `colour`, `size` required; `top_n` needs `orderby` and `orderby_ascending`; `filter_fn`
  keeps an interaction for every pair if any of its rows passes; absent labels raise.
- `tileplot`: `fill`/`label` are suffixes existing as both `ligand_<x>` and `receptor_<x>` (`means`,
  `props`, a `df_to_lr` stat). CellChat has no `*_means`: use `fill="trimean"`.
- `circle`: cell-type network, `pivot_mode="counts"` or `"mean"`.
- `dotplot_by_sample(adata, colour=..., size=...)` facets by sample; subset labels or it explodes.
- Spatial results (`bivariate`, `inflow`) are AnnData: `sc.pl.spatial(lrdata, color="LIG^REC")` or
  `sc.pl.embedding(lrdata, basis="spatial", color=...)`.
