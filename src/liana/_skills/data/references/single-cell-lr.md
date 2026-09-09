# Single-cell ligand-receptor inference (dissociated data)

Aim: rank LR interactions between cell types in one dataset. Tutorial: `basic_usage`.

## Call

```python
import liana as li
li.mt.rank_aggregate(adata, groupby="cell_type")
adata.uns["liana_res"]   # long DataFrame, sorted by magnitude
li.pl.dotplot(adata, colour="magnitude_rank", size="specificity_rank", inverse_colour=True, inverse_size=True, top_n=20)
```

Always show the top hits with `li.pl.dotplot` (arguments in `outputs-and-plotting.md`).

`li.mt.cellphonedb`, `cellchat`, `natmi`, `connectome`, `singlecellsignalr`, `logfc`,
`geometric_mean`, `scseqcomm` share this exact signature. `li.mt.show_methods()` lists them with
their score columns and references. `rank_aggregate` aggregates five of them (CellPhoneDB,
Connectome, log2FC, NATMI, SingleCellSignalR) into `magnitude_rank` and `specificity_rank`, both
"lower is better", and keeps every sub-method's own columns.

| arg | default | note |
|---|---|---|
| `groupby` | required | `obs` column with cell types; groups with fewer than `min_cells=5` cells are dropped with a warning |
| `expr_prop` | 0.05 | LR pairs whose ligand or receptor (any subunit) is expressed in fewer cells of the group are dropped. `return_all_lrs=True` keeps them with the worst observed score |
| `n_perms` | 1000 | permutations for p-values. `None` skips them: no p-value columns, and `rank_aggregate` returns `magnitude_rank` only (asking for specificity then raises) |
| `use_raw`, `layer` | False, None | which matrix; log1p-normalised, non-negative |
| `resource_name`, `resource`, `interactions` | consensus | see prior-knowledge.md; `interactions=[("L","R"), ...]` overrides both others |
| `groupby_pairs` | None | DataFrame with `source`, `target` columns: score only those cell-type pairs |
| `key_added`, `inplace` | "liana_res", True | `inplace=False` returns the DataFrame |

Magnitude scores answer "how strongly expressed is this pair", specificity scores "how specific is it
to this pair of cell types". Read `outputs-and-plotting.md` for the per-method columns and the plots.

## Footguns

- Scaled data (or `.raw` holding scaled data) gives NaN `lr_logfc`, NaN `specificity_rank`, and
  downstream `circle_plot` KeyErrors. Check `adata.X.min() >= 0` first.
- Missing dots in a dotplot are usually pairs filtered by `expr_prop`, not a bug.
- The toy `li.ds.generate_toy_adata()` holds only variable genes, so a high "missing resource
  elements" fraction is expected there.
- A custom set of methods: `li.mt.AggregateClass(li.mt.aggregate_meta, methods=[li.mt.logfc, li.mt.geometric_mean])`.

## Variants

- **Coordinates present (spatial single-cell data)**: constrain scores by cell-type proximity with
  `spatial_key="spatial", spatial_kwargs={"bandwidth": 100}` on any of these
  methods. Bandwidth is in coordinate units: 100 µm is the usual diffusion assumption, so convert
  pixels first (the pixel size is normally in `adata.uns`). Scores are multiplied by the pair's
  proximity from `li.pp.spatial_pair_proximity`, and pairs with fewer than
  `min_cells_in_proximity=10` close cell pairs are set to 0. For the per-cell alternative read `inflow.md`.
- **Restrict to co-localised pairs**: pass `groupby_pairs` built from MISTy contributions, global
  Moran's R, or squidpy neighbourhood enrichment.
- **Several samples**: `li.mt.rank_aggregate.by_sample(adata, sample_key="sample", ...)` adds a
  sample column; read `multisample.md` for what to do with it.
- **MuData (e.g. CITE-seq)**: `li.mt.rank_aggregate(mdata, groupby="cell_type", resource=res,
  mdata_kwargs={"x_mod": "rna", "y_mod": "prot", "x_transform": li.pp.zi_minmax, "y_transform": li.pp.zi_minmax})`.
  `x_mod` supplies ligands, `y_mod` receptors; `groupby` must be in `mdata.obs`; top-level
  `use_raw`/`layer` are ignored, use `x_use_raw`/`x_layer`. Modalities must be non-negative, hence
  the transform. Identical feature names across modalities collide: prefix one side (e.g. `"AB:"`)
  in both the modality and the resource. Protein `var_names` must be HGNC symbols to match the resource
  (map antigen names such as `CD8a`, `PD-1` yourself), and must not contain `_`, which marks complex
  subunits in the resource: strip suffixes like `_TotalSeqB` first.
- **Metabolite-mediated interactions from RNA alone**: estimate metabolite abundance with MetalinksDB and
  score metabolite-receptor pairs with this same call; read `metabolites.md`.
