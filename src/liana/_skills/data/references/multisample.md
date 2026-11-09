# Several samples or conditions (dissociated data)

Comparing per-condition `rank_aggregate` tables is indicative only (no replicates); with one sample per
condition nothing can be tested: say so. With replicates, name the four routes in one line and say why
you picked one. Cell groups must be present across samples.

1. Targeted: pseudobulk DE per cell type, mapped onto LR pairs (clear two-group contrast).
2. MOFA+: unsupervised factors of LR variation across samples (many samples, unknown structure).
3. Tensor-cell2cell: sample x LR x sender x receiver decomposition (context-dependent programs).
4. pyCrossTalkeR (external): network differences between per-condition tables.

## Per-sample scores (routes 2-3)

```python
li.mt.rank_aggregate.by_sample(adata, sample_key="sample", groupby="cell_type",
                               n_perms=None, return_all_lrs=True)
li.pl.dotplot_by_sample(adata, colour="magnitude_rank", size="specificity_rank",
                        inverse_colour=True, inverse_size=True, source_labels=sources, target_labels=targets)
```

Only single-cell methods have `.by_sample`. Drop samples whose cells all fall in one `groupby` level (`ValueError`).

## 1. Targeted: pseudobulk DE -> LR (tutorial `targeted`)

<!-- skip-exec: extras -->
```python
import decoupler as dc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

pdata = dc.pp.pseudobulk(adata, sample_col="sample", groups_col="cell_type", layer="counts", mode="sum")
dea = {}
for ct in pdata.obs["cell_type"].unique():
    ctdata = pdata[pdata.obs["cell_type"] == ct].copy()
    ctdata = ctdata[:, dc.pp.filter_by_expr(ctdata, group="condition", inplace=False)].copy()
    dds = DeseqDataSet(adata=ctdata, design="~condition")
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", "stim", "ctrl"])
    stats.summary()
    dea[ct] = stats.results_df
dea_df = pd.concat(dea, names=["cell_type"]).reset_index("cell_type")
lr = li.mt.df_to_lr(adata, dea_df, groupby="cell_type", stat_keys=["stat", "pvalue", "padj"], complex_col="stat")
li.pl.tileplot(liana_res=lr, fill="stat", label="padj", label_fn=lambda p: "*" if p < 0.05 else "",
               top_n=15, orderby="interaction_stat")
```

- `layer` must hold raw counts. With more than two conditions pick one contrast per run and say which.
- `adata` passed to `df_to_lr` supplies expression proportions (subset it to one condition if wanted).
- `interaction_<stat>` is the mean of `ligand_<stat>` and `receptor_<stat>`: judge both sides.
  `complex_col` picks the subunit by absolute minimum, so not for p-values. pydeseq2 leaves some NaN p-values.

Causal network from receptors (their DE stats) to TFs (activities, e.g. `dc.mt.ulm` on the DE stats);
`node_weights` in [0, 1] (expression proportions).
Leave `solver` unset: CORNETO picks GUROBI/CPLEX/COPT/MOSEK if installed, else SciPy's HiGHS, then
SCIP (`pyscipopt`, in extras).

<!-- skip-exec: extras -->
```python
G = li.rs.build_prior_network(ppis, input_nodes=receptor_scores, output_nodes=tf_scores)   # ppis: source, mor, target
net_df, problem = li.mt.find_causalnet(G, receptor_scores, tf_scores, node_weights=expr_props,
                                       edge_penalty=0.1, max_runs=10, stable_runs=50)
```

## 2. MOFA+ (tutorial `mofatalk`; gene-level: `mofacellular`, `li.ms.adata_to_views`)

<!-- skip-exec: extras -->
```python
import muon
mdata = li.ms.lrs_to_views(adata, score_key="magnitude_rank", obs_keys=["condition"], lr_prop=0.3)
muon.tl.mofa(mdata, use_obs="union", n_factors=3, convergence_mode="medium", outfile="mofa.h5ad")
scores = li.ms.get_factor_scores(mdata, obsm_key="X_mofa", obs_keys=["condition"])
loadings = li.ms.get_variable_loadings(mdata, varm_key="LFs", view_sep=":", pair_sep="&", variable_sep="^")
```

Views are `source&target` pairs, variables `ligand^receptor`. `lr_fill`: `np.nan` (default) lets MOFA
impute absent interactions, `0` declares absence meaningful; expose that choice. `obs_keys` must be
constant within a sample. Loading signs are relative to the factor score sign.

## 3. Tensor-cell2cell (tutorial `liana_c2c`; https://ccc-protocols.readthedocs.io)

<!-- skip-exec: extras -->
```python
import cell2cell as c2c
tensor = li.ms.to_tensor_c2c(adata, sample_key="sample", score_key="magnitude_rank", how="outer_cells")
context = dict(adata.obs[["sample", "condition"]].drop_duplicates().to_numpy())
meta = c2c.tensor.generate_tensor_metadata(interaction_tensor=tensor, metadata_dicts=[context, None, None, None],
                                           fill_with_order_elements=True)
tensor = c2c.analysis.run_tensor_cell2cell_pipeline(tensor, meta, rank=None, device="cpu")
```

`how`, `lr_fill`, `cell_fill` shape the tensor; `rank=None` (elbow) is slow without a GPU.

## 4. pyCrossTalkeR

Per-condition tables from `by_sample(adata, sample_key="condition")` go to `pycrosstalker` (`cttl.utils.from_liana`).
