# Spatial setup: units, bandwidth, graph

Every distance in liana (`bandwidth`, `max_radius`, `radius_step`, `spatial_kwargs`) is in the units
of `obsm["spatial"]`. Think in µm, then convert the distance to those units. `bivariate`, `inflow` and
`lric` have no default resource: pass `resource_name=` (or `resource`/`interactions`).

## Units

- Already µm: Xenium, MERSCOPE, spatialdata exports. Pixels: Visium via scanpy/squidpy, raw CosMx.
- Visium: µm per full-res pixel = `sf.get("microns_per_pixel") or 55 / sf["spot_diameter_fullres"]`,
  (a spot is 55 µm wide, centres 100 µm apart). spatialdata: the `Scale` transformation to `global`.

```python
library_id = next(iter(adata.uns["spatial"]))
sf = adata.uns["spatial"][library_id]["scalefactors"]
um_per_px = sf.get("microns_per_pixel") or 55 / sf["spot_diameter_fullres"]
bandwidth = 100 / um_per_px          # 100 µm in pixels; leave obsm["spatial"] in pixels for sc.pl.spatial
```

## Bandwidth

```python
plot, nn = li.pp.query_bandwidth(coordinates=adata.obsm["spatial"])   # neighbours within each distance
li.pp.spatial_neighbors(adata, bandwidth=bandwidth, set_diag=True)     # spots; cells: drop set_diag
```

- Sanity check (in µm): nearest neighbours ~100 µm apart for Visium spots, 5-20 µm for cells.
- Anchors: ~100 µm (ligand diffusion), or the radius admitting the first ring of 6 Visium
  neighbours, or 10-20 cells for single-cell data. Too wide blurs local patterns, too narrow misses
  gradients. Report the distance and the chosen bandwidth before scoring.
- `set_diag=True` for spots (a spot holds both partners), `False` (default) for single cells.
- All spatial methods read `obsp["spatial_connectivities"]` (squidpy's graph works too). A stale graph is reused silently: re-run after changing coordinates,
  bandwidth or the cell subset.
- Several slides in one object share coordinate space: `adata = li.pp.expand_coordinates(adata,
  sample_key="sample")` (returns a copy) before `spatial_neighbors`.
