# Metabolite-mediated communication

Two sources of metabolite abundance: **estimated** from transcriptomics via MetalinksDB (Farr et al.,
Brief Bioinform 2024, doi:10.1093/bib/bbae347; enzyme and transporter expression), or **measured**
(spatial metabolomics, MSI). Both become a
modality in a MuData and then run through the usual methods. Tutorials: `sc_multi` (estimated),
`sma` (measured).

## Estimated from RNA

```python
ml = li.rs.get_metalinks(tissue_location="Brain", biospecimen_location=["Blood", "Cerebrospinal Fluid"],
                         source=["CellPhoneDB", "NeuronChat"], types=["lr", "pd"])   # downloads metalinksdb.db to cwd
resource = ml[ml["type"] == "lr"][["metabolite", "gene_symbol"]].rename(columns={"metabolite": "source", "gene_symbol": "receptor"})
pd_net = (ml[ml["type"] == "pd"].groupby(["metabolite", "gene_symbol"])["mor"].mean().reset_index()
            .rename(columns={"metabolite": "source", "gene_symbol": "target", "mor": "weight"}))
t_net = ml[ml["type"] == "pd"][["metabolite", "gene_symbol", "transport_direction"]].dropna()   # optional transporters
t_net["weight"] = t_net["transport_direction"].map({"out": 1, "in": -1})                     # export +1, import -1
t_net = (t_net.groupby(["metabolite", "gene_symbol"])["weight"].mean().reset_index().query("weight != 0")
           .rename(columns={"metabolite": "source", "gene_symbol": "target"}))
meta = li.mt.estimate_metalinks(adata, resource, pd_net=pd_net, t_net=t_net, tmin=3)
meta.obs["cell_type"] = adata.obs["cell_type"]                     # obs lives on the MuData container
li.mt.rank_aggregate(meta, groupby="cell_type", resource=resource.rename(columns={"source": "ligand"}),
                     mdata_kwargs={"x_mod": "metabolite", "y_mod": "receptor",
                                   "x_transform": li.pp.zi_minmax, "y_transform": li.pp.zi_minmax})
```

- `get_metalinks` filters are AND-combined; `li.rs.get_metalinks_values(table, column)` lists the
  allowed values, `li.rs.describe_metalinks()` the schema. Needs `requests`.
- `estimate_metalinks` needs `decoupler>=2`; extra kwargs (e.g. `tmin`) go to both the enzyme and
  transporter steps. It returns a MuData with modalities `metabolite` (signed activity scores) and
  `receptor`. Metabolites without a transporter entry are left unmasked.
- The metabolite scores are signed, so the `zi_minmax` transform is required before the
  single-cell methods. Spatial data: `li.mt.bivariate(meta, x_mod="metabolite", y_mod="receptor", ...)`.
- State the caveat: this assumes a linear link between enzyme expression and metabolite abundance
  and treats metabolites independently, so results are hypotheses.
- Non-human data: translate `gene_symbol` with `li.rs.translate_column` (prior-knowledge.md).

## Measured (spatial metabolomics + transcriptomics)

```python
mdata = mu.MuData({"rna": rna, "msi": msi})                        # separate AnnData per modality
msi_on_spots = li.pp.interpolate_adata(target=msi, reference=rna, spatial_key="spatial")  # if grids differ
li.pp.spatial_neighbors(mdata, bandwidth=bandwidth, set_diag=True)    # spots: include self
interactions = ml[["metabolite", "gene_symbol"]].apply(tuple, axis=1).tolist()
lr = li.mt.bivariate(mdata, x_mod="msi", y_mod="rna", interactions=interactions,
                     x_transform=sc.pp.scale, y_transform=sc.pp.scale, mask_negatives=True, n_perms=100)
```

Metabolite and gene names must match the `var_names` of their modality. For MISTy, pass the
modalities as views: `li.mt.MistyData({"intra": msi, "receptor": rec}, enforce_obs=False, obs=mdata.obs)`
with per-view connectivities built against the MSI coordinates (`reference=`).
