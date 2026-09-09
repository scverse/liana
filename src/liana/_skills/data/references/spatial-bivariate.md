# Spatial bivariate metrics (local and global)

Aim: where in the tissue do a ligand and a receptor co-vary, per spot or cell (local) and slide-wide
(global). Works on spots (Visium) and cells. Tutorial: `bivariate`. A spot holds several cells, so on
Visium this answers where ligand and receptor co-occur, not which cell types interact; for that use
deconvolution proportions with `inflow` (`inflow.md`, `obsm_key`).

## 1. Spatial graph

```python
p, df = li.pp.query_bandwidth(coordinates=adata.obsm["spatial"])
li.pp.spatial_neighbors(adata, bandwidth=200, set_diag=True)     # spots: include self
adata.obsp["spatial_connectivities"]     # what every spatial method reads
```

- `bandwidth` is required and is in the units of `obsm["spatial"]`; every distance in liana is. Get
  coordinates into µm first. Xenium, MERSCOPE and spatialdata exports are already µm; Visium via
  scanpy/squidpy and raw CosMx are pixels. Multiply by the µm-per-pixel ratio stored in the object:
  Visium `sf = adata.uns["spatial"][lib]["scalefactors"]`, ratio `sf.get("microns_per_pixel") or
  55 / sf["spot_diameter_fullres"]` (a spot is 55 µm wide, adjacent centres 100 µm apart); spatialdata
  objects carry it as the `Scale` transformation to the `global` coordinate system; otherwise the
  instrument metadata. Check with `query_bandwidth`: the nearest-neighbour distance should be about
  100 for Visium spots and 5 to 20 for cells; a value in the hundreds for cells means pixels. Report
  that distance and the chosen bandwidth before scoring. Rules of thumb: about 100 µm (ligand
  diffusion), or the radius that admits the 6 nearest spots (first Visium ring), or 10 to 20 cells
  for single-cell data.
- `set_diag=True` for spots (a spot is its own neighbour), `False` for single cells.
- `max_neighbours=100` caps the k-NN search; lower it for very large slides.
- Squidpy's `sq.gr.spatial_neighbors` output works too, under the same `obsp` key.

## 2. Score

```python
lr = li.mt.bivariate(adata, global_name="morans", n_perms=100, add_categories=True)
```

Returns a **new** AnnData (obs = spots, var = interactions named `LIG^REC`):

- `.X` local scores; `.layers["pvals"]` if `n_perms` is set; `.layers["cats"]` if `add_categories`
  (1 high-high, -1 high-low, 0 neither).
- `.var` global stats: `morans`, `morans_pvals` (and/or `lee`), plus `ligand_means`, `ligand_props`,
  `receptor_*`, `mean`, `std`.
- `.obs`, `.obsm`, `.obsp` copied, so `sc.pl.spatial(lr, color="LIG^REC")` works directly.
- `local_name=None` returns only the global DataFrame.

`local_name`: `cosine` (default, best in the LIANA+ benchmark), `jaccard` (binary-like data),
`pearson`, `spearman`, `masked_spearman`, `morans`, `product`, `norm_product`;
`li.mt.bivariate.show_functions()` lists them. `global_name`: `"morans"`, `"lee"`, or a list.

`n_perms`: `None` no p-values; `>0` permutation p-values (the slow part); `0` analytical p-values,
Moran's R only. `mask_negatives=True` zeroes scores and sets p to 1 wherever the category is not
high-high. `use_raw` and `layer` are keyword-only extras, not positional.

## Footguns

- Missing `obsp["spatial_connectivities"]` raises: always run `spatial_neighbors` first.
- Several slides concatenated share coordinate space and produce false neighbours: run
  `li.pp.expand_coordinates(adata, sample_key="sample")` before `spatial_neighbors`.
- Moran's R is limited to two variables; for non-linear or multi-factor questions use MISTy (`misty.md`).
- Local scores can be summarised into patterns with `li.ms.nmf(lr, n_components=None)` (needs
  `kneed`; only for non-negative metrics); show the rank choice with `li.pl.elbow(lr)`.

## Variants

- **MuData** (e.g. cell-type proportions vs TF activities, RNA vs metabolites):
  `li.mt.bivariate(mdata, x_mod="rna", y_mod="msi", x_use_raw=False, y_use_raw=False,
  x_transform=sc.pp.scale, y_transform=sc.pp.scale, interactions=[(x, y), ...], mask_negatives=True)`.
  The connectivities must be on the MuData (`li.pp.spatial_neighbors(mdata, ...)` works).
  With MuData the resource columns default to `x` and `y`, not `ligand`/`receptor`, so pass
  `interactions` or rename the resource columns.
- **Unaligned modalities** (different coordinates): `li.pp.interpolate_adata(target=msi,
  reference=rna, spatial_key="spatial")` regrids one onto the other, or
  `li.pp.spatial_neighbors(other, reference=ref_coords, ...)` writes a rectangular graph to `.obsm`.
- **Deconvolution proportions or activity scores** live in `.obsm`; lift them with
  `li.pp.obsm_to_adata(adata, "cell2location")` to use as a modality.
- **Metabolites**: measured (MSI) as a modality, or estimated from RNA via MetalinksDB; read `metabolites.md`.
