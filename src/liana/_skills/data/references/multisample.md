# Several samples or conditions (dissociated data)

Comparing per-condition `rank_aggregate` tables is only indicative: it has no notion of replicates.
With one sample per condition nothing can be tested; say so. With replicates, three routes.
All take an AnnData; MuData is built internally where needed. Cell groups must be well defined and
present across samples. Name the three routes to the user in one line and say why you picked one.

## 0. Per-sample scores (shared first step for routes 2 and 3)

```python
li.mt.rank_aggregate.by_sample(adata, sample_key="sample", groupby="cell_type",
                               n_perms=None, return_all_lrs=True)
adata.uns["liana_res"]          # one block per sample, extra column "sample"
li.pl.dotplot_by_sample(adata, colour="magnitude_rank", size="specificity_rank",
                        inverse_colour=True, inverse_size=True, source_labels=sources, target_labels=targets)
```

Any method has `.by_sample`. A sample whose cells all fall in one `groupby` level raises a
ZeroDivisionError in the log-fold-change step; drop such samples.

## 1. Targeted differential: pseudobulk DE, then LR table, then optional causal network

Tutorial `targeted`. Needs `decoupler>=2`, `pydeseq2`; causal network needs `corneto` (all in `liana[extras]`).

```python
pdata = dc.pp.pseudobulk(adata, sample_col="sample", groups_col="cell_type", layer="counts", mode="sum")
dea_df = ...  # one pydeseq2 result table per cell type, concatenated, with a "cell_type" column
adata_cond = adata[adata.obs["condition"] == "treated"]                 # cells of the contrast's condition of interest
lr = li.mt.df_to_lr(adata_cond, dea_df, groupby="cell_type", stat_keys=["stat", "pvalue", "padj"], complex_col="stat")
li.pl.tileplot(liana_res=lr, fill="stat", label="padj", label_fn=lambda x: "*" if x < 0.05 else "",
               top_n=15, orderby="interaction_stat")
```

- Pseudobulk and DE are not liana's: follow decoupler (`dc.pp.pseudobulk`, `dc.pp.filter_by_expr`)
  and pydeseq2 (`DeseqDataSet(design="~condition")`, `DeseqStats(contrast=[...])`) docs, running DE
  per cell type. `layer` must hold raw counts; check `layers` first. With more than two conditions
  pick one contrast per run and tell the user which.
- `dea_df` needs the DE statistics as columns (`stat_keys`), gene names as index matching
  `adata.var_names`, and the `groupby` column; `adata_cond` supplies expression proportions.
- Output has `ligand_<stat>`, `receptor_<stat>` per stat, and `interaction_<stat>` which is just
  their mean: filter and plot on the ligand and receptor statistics separately.
- `complex_col` picks the subunit by absolute minimum, so not suitable for p-values.
- pydeseq2 leaves NaN p-values for some genes (Cook's and independent filtering); check them.

Causal network from receptors to TFs:

```python
G = li.rs.build_prior_network(ppis, input_nodes=receptor_scores, output_nodes=tf_scores)   # ppis: source, mor, target
df, P = li.mt.find_causalnet(G, receptor_scores, tf_scores, node_weights=expr_props,
                             edge_penalty=0.1, max_runs=10, stable_runs=50)
```

`node_weights` must lie in [0, 1] (expression proportions in the target cell type). Leave `solver`
unset: CORNETO picks the SciPy HiGHS backend, which needs nothing beyond `liana[extras]`; named
solvers need their own package (`GUROBI`, `SCIP`, `HIGHS` via highspy). Visualise with
`corneto.methods.carnival.visualize_network(df)`.

## 2. MOFA+: factors of LR variation across samples

Tutorial `mofatalk` (`mofacellular` for the gene-level variant via `li.ms.adata_to_views`).
Needs `mofapy2`, `mofax`, `muon`.

```python
mdata = li.ms.lrs_to_views(adata, score_key="magnitude_rank", obs_keys=["condition"],
                           lr_prop=0.3, lrs_per_sample=20, samples_per_view=5)
mu.tl.mofa(mdata, use_obs="union", n_factors=5, convergence_mode="medium", seed=1337, outfile="mofa.h5ad")
scores = li.ms.get_factor_scores(mdata, obsm_key="X_mofa", obs_keys=["condition"])
loads = li.ms.get_variable_loadings(mdata, varm_key="LFs", view_sep=":", pair_sep="&", variable_sep="^")
```

- Views are `source&target` cell-type pairs, variables `ligand^receptor`; ascending scores are
  inverted with `-log10` internally.
- `lr_fill` is a key choice: `np.nan` (default) lets MOFA impute interactions absent in a sample;
  `0` states that absence is biologically meaningful. Expose it to the user.
- `obs_keys` must be constant within a sample. Loading signs are relative to the factor score sign.

## 3. Tensor-cell2cell: sample x LR x sender x receiver decomposition

Tutorial `liana_c2c`; in-depth material at https://ccc-protocols.readthedocs.io. Needs `cell2cell`
(`tensorly`, `torch` for GPU).

```python
tensor = li.ms.to_tensor_c2c(adata, sample_key="sample", score_key="magnitude_rank", how="outer_cells")
# then external: c2c.tensor.generate_tensor_metadata(...) and c2c.analysis.run_tensor_cell2cell_pipeline(tensor, meta, rank=None, ...)
```

Run `by_sample` with `return_all_lrs=True` first so non-expressed pairs are known. Duplicate
(sample, source, target, LR) rows raise. `how`, `lr_fill`, `cell_fill` shape the tensor. Rank
estimation (`rank=None`) is slow without a GPU.

## 4. pyCrossTalkeR (external)

liana only supplies per-condition tables via `li.mt.<method>.by_sample(adata, sample_key="condition")`;
`pycrosstalker` (`cttl.utils.from_liana`, `cttl.analise_LR`) does the network differential.
