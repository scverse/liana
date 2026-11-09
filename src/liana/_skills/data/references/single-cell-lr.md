# Single-cell ligand-receptor inference (dissociated data)

Aim: rank LR interactions between cell types in one dataset. Tutorial: `basic_usage`.

```python
li.mt.rank_aggregate(adata, groupby="cell_type", expr_prop=0.1)
adata.uns["liana_res"].head()
```

- `rank_aggregate` (default choice) combines CellPhoneDB, Connectome, log2FC, NATMI and
  SingleCellSignalR into `magnitude_rank` ("how strongly expressed") and `specificity_rank`
  ("how specific to this cell-type pair"). Use one method when the user names it or a resource-specific
  model matters.
- `li.mt.cellphonedb`, `cellchat`, `natmi`, `connectome`, `singlecellsignalr`, `logfc`,
  `geometric_mean`, `scseqcomm` share the same call; `li.mt.show_methods()` lists them.
- Custom aggregate: `li.mt.AggregateClass(li.mt.aggregate_meta, methods=[li.mt.logfc, li.mt.geometric_mean])`.

| arg | decision |
|---|---|
| `groupby` | cell-type column; groups under `min_cells` (5) cells are dropped with a warning |
| `expr_prop` | raise for noisy data; `return_all_lrs=True` keeps filtered pairs with the worst score |
| `n_perms` | `None` skips permutations: no p-value columns, faster |
| `groupby_pairs` | DataFrame (`source`, `target`): score only these pairs, e.g. co-localised ones (MISTy, Moran's R) |
| `inplace=False` | return the DataFrame instead of writing `uns` |

## Variants

- **Coordinates present** (spatial single cells): `spatial_key="spatial", spatial_kwargs={"bandwidth": 100}`
  multiplies scores by cell-type pair proximity (`li.pp.spatial_pair_proximity`); pairs with fewer than
  `min_cells_in_proximity` (10) close cell pairs score 0. Cheaper than per-cell Inflow.
- **MuData (CITE-seq)**: `x_mod` supplies ligands, `y_mod` receptors; `groupby` must be in `mdata.obs`;
  top-level `use_raw`/`layer` are ignored (use `x_layer`, `y_use_raw`, ...). Protein `var_names` must
  be HGNC symbols (map antigen names such as `PD-1` yourself) without `_` (strip `_TotalSeqB`), and
  prefixed in both modality and resource so they do not collide with RNA names:

```python
mdata.obs["cell_type"] = mdata.mod["rna"].obs["cell_type"]
resource = li.rs.select_resource()
resource["receptor"] = "AB:" + resource["receptor"].str.replace("_", "_AB:")   # prefix every subunit
mdata.mod["prot"].var_names = "AB:" + mdata.mod["prot"].var_names
li.mt.rank_aggregate(mdata, groupby="cell_type", resource=resource,
                     mdata_kwargs={"x_mod": "rna", "y_mod": "prot",
                                   "x_transform": li.pp.zi_minmax, "y_transform": li.pp.zi_minmax})
```
