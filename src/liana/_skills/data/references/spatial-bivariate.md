# Spatial bivariate metrics (local and global)

Aim: where in the tissue do a ligand and a receptor co-vary, per spot or cell (local) and slide-wide
(global). Tutorial: `bivariate`. On Visium a spot holds several cells, so this answers where ligand and
receptor co-occur, not which cell types interact (for that: Inflow with deconvolution proportions).

```python
lrdata = li.mt.bivariate(adata, resource_name="consensus", global_name="morans", n_perms=100, add_categories=True)
```

Returns a **new** AnnData (obs = spots, var = `LIG^REC`); `local_name=None` returns only the global DataFrame.

- `.X` local scores; `.layers["pvals"]` with `n_perms`; `.layers["cats"]` with `add_categories`
  (1 high-high, -1 high-low or low-high, 0 low-low).
- `.var`: global stats (`morans`, `morans_pvals`, and/or `lee`), `ligand_means`, `ligand_props`, `receptor_*`.
- `.obs`, `.obsm`, `.obsp` copied, so spatial plotting works directly.

Choices:
- `local_name`: `cosine` (default, best in the LIANA+ benchmark), `jaccard` (binary-like data),
  `pearson`, `spearman`, `masked_spearman`, `morans`, `product`, `norm_product`; `li.mt.bivariate.show_functions()`.
- `n_perms`: `None` no p-values; `>0` permutations (the slow part); `0` analytical, Moran's R only.
- `mask_negatives=True` zeroes scores and sets p to 1 where the category is not high-high.
- Moran's R is bivariate and linear; for non-linear or multi-feature questions use MISTy.
- Patterns of local scores: `li.ms.nmf(lrdata, n_components=None)` (needs `kneed`; non-negative
  metrics only); show the rank choice with `li.pl.elbow`.

## Variants

- **MuData** (cell-type proportions vs TF activities, RNA vs metabolites): `x_mod`/`y_mod`, per-side
  `x_use_raw`, `x_layer`, `x_transform`; connectivities on the MuData. Resource columns default to `x`
  and `y`, so pass `interactions`:

```python
lrdata = li.mt.bivariate(mdata, x_mod="adata_x", y_mod="adata_y", interactions=interactions,
                         x_transform=sc.pp.scale, y_transform=sc.pp.scale, mask_negatives=True)
```

- **Unaligned modalities**: `li.pp.interpolate_adata` regrids one onto the other.
- **Deconvolution proportions or activity scores** in `.obsm`: lift with
  `li.pp.obsm_to_adata(adata, "cell2location")` to use as a modality.
