# Metabolite-mediated communication

Metabolite abundance is either **estimated** from enzyme and transporter expression via MetalinksDB
(Farr et al., Brief Bioinform 2024, doi:10.1093/bib/bbae347) or **measured** (spatial metabolomics,
MSI). Either becomes a modality in a MuData scored by the usual methods. Tutorials: `sc_multi`
(estimated), `sma` (measured).

## Estimated from RNA

<!-- skip-exec: network -->
```python
ml = li.rs.get_metalinks(tissue_location="Brain", biospecimen_location=["Blood", "Cerebrospinal Fluid"],
                         types=["lr", "pd"])
resource = ml[ml["type"] == "lr"][["metabolite", "gene_symbol"]].rename(columns={"metabolite": "source", "gene_symbol": "receptor"})
pd_net = (ml[ml["type"] == "pd"].groupby(["metabolite", "gene_symbol"])["mor"].mean().reset_index()
            .rename(columns={"metabolite": "source", "gene_symbol": "target", "mor": "weight"}))
t_net = ml[ml["type"] == "pd"][["metabolite", "gene_symbol", "transport_direction"]].dropna()
t_net["weight"] = t_net["transport_direction"].map({"out": 1, "in": -1})       # export +1, import -1
t_net = (t_net.groupby(["metabolite", "gene_symbol"])["weight"].mean().reset_index().query("weight != 0")
           .rename(columns={"metabolite": "source", "gene_symbol": "target"}))
meta = li.mt.estimate_metalinks(adata, resource, pd_net=pd_net, t_net=t_net, tmin=3)
meta.obs["cell_type"] = adata.obs["cell_type"]
li.mt.rank_aggregate(meta, groupby="cell_type", resource=resource.rename(columns={"source": "ligand"}),
                     mdata_kwargs={"x_mod": "metabolite", "y_mod": "receptor",
                                   "x_transform": li.pp.zi_minmax, "y_transform": li.pp.zi_minmax})
```

- `get_metalinks` filters are AND-combined; `source=` filters both lr and pd curators, so keep a
  production-degradation source (rhea, recon, hmr) if you set it. `li.rs.get_metalinks_values(table,
  column)` lists allowed values; `db_path=` reads a local copy.
- Rows are one per edge and curating source, so the mean `mor` is a vote: an edge annotated both
  produced (+1) and degraded (-1) cancels out.
- `estimate_metalinks` (needs `decoupler>=2`, AnnData only) passes extra kwargs (`tmin`) to both enzyme
  and transporter steps; returns modalities `metabolite` (signed scores) and `receptor`. Signed
  scores need `zi_minmax` before single-cell methods. Spatial data: `li.mt.bivariate(meta,
  x_mod="metabolite", y_mod="receptor", ...)`.
- Caveat to state: assumes a linear enzyme-abundance link and independent metabolites; results are hypotheses.
- MetalinksDB gene symbols are human: for non-human data translate `gene_symbol` with
  `li.rs.translate_column` (HCOP orthologs) before building the resource and networks.

## Measured (MSI + transcriptomics)

<!-- skip-exec: network -->
```python
msi = li.pp.interpolate_adata(target=msi, reference=rna, spatial_key="spatial")   # only if grids differ
mdata = mu.MuData({"rna": rna, "msi": msi})
mdata.obsm["spatial"] = rna.obsm["spatial"]
li.pp.spatial_neighbors(mdata, bandwidth=100, set_diag=True)
interactions = list(resource.itertuples(index=False, name=None))   # (metabolite, receptor) pairs
lrdata = li.mt.bivariate(mdata, x_mod="msi", y_mod="rna", interactions=interactions,
                         x_transform=sc.pp.scale, y_transform=sc.pp.scale, mask_negatives=True, n_perms=100)
```

Metabolite names must match the MSI `var_names`. For MISTy, pass modalities as views:
`li.mt.MistyData({"intra": msi, "receptor": rna}, enforce_obs=False, obs=mdata.obs)` with per-view
connectivities built against the MSI coordinates (`reference=`).
