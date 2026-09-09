---
name: liana
description: Cell-cell communication (CCC) inference with the liana Python package (LIANA+, scverse). Use for any task involving liana or ligand-receptor (LR) analysis of AnnData/MuData objects. Triggers on steady-state LR scoring (rank_aggregate, CellPhoneDB, CellChat, NATMI, Connectome, SingleCellSignalR, logFC, scSeqComm); multi-sample or differential CCC (by_sample, MOFA+, Tensor-cell2cell, df_to_lr, pyCrossTalkeR); spatial CCC on Visium, Xenium, MERFISH, CosMx or slide-seq (spatial_neighbors, bivariate local/global metrics, Moran's R, Inflow, LRIC, cross-PCF, MISTy); multimodal CITE-seq or spatial metabolomics; metabolite-mediated CCC via MetalinksDB; LR resources and orthology for mouse or other organisms (consensus, mouseconsensus, OmniPath, HCOP); liana plots (dotplot, tileplot, circle_plot). Also use when the user says cell-cell interactions, crosstalk, signalling between cell types, sender and receiver, or niche signalling, even without naming liana. Not for MOFA+, MISTy or Tensor-cell2cell used outside liana.
---

# liana (LIANA+)

`import liana as li`. Submodules: `li.mt` methods, `li.pp` preprocessing (spatial graphs,
transforms), `li.rs` prior knowledge, `li.pl` plots, `li.ms` multi-sample helpers, `li.ds` datasets.
Every method accepts an `AnnData` or a `MuData`. Snippets also assume `import scanpy as sc, mudata as mu,
decoupler as dc, numpy as np, pandas as pd`.

## Workflow

1. **Intake.** Ask for the object or a path to it, and say explicitly that a description of the
   data is fine instead if it cannot be shared (patient data, privacy). Raw vendor output is not an
   object yet: a Space Ranger `outs/` folder loads with `sc.read_visium`, a 10x `.h5` with
   `sc.read_10x_h5`, Xenium, CosMx or MERSCOPE exports with `spatialdata_io`. If a path is given,
   run the snippet below and infer what you can; then ask only what the object cannot tell you,
   usually the aim and, when several columns qualify, which one holds the condition or cell type.
   Without an object, ask in plain words and only what the message left open: tissue and
   technology; one or several samples and how they group; whether cell types are annotated;
   species; any modality beyond RNA (protein, measured metabolites); and what they want to learn.
   Ask about coordinate units only for spatial data, and whether counts were normalised and
   log-transformed (e.g. scanpy `normalize_total` + `log1p`) only if the snippet could not run.
2. **Select** the branch from the table.
3. **Read** that one reference file. Also read `prior-knowledge.md` for non-human data or a custom
   LR list, and `outputs-and-plotting.md` before plotting or interpreting `liana_res`.
4. **Run**, then explain in one paragraph without column names: what was scored, which method and
   why, what a high score means, and the main caveat. Name any bandwidth or threshold you chose.

```python
import anndata as ad, mudata as mu, numpy as np

a = mu.read(path) if path.endswith(".h5mu") else ad.read_h5ad(path)
print(type(a).__name__, a.shape, "mods:", list(getattr(a, "mod", {})))
for c in a.obs.columns:
    if a.obs[c].dtype.kind not in "biuf":
        u = a.obs[c].unique()
        print(c, len(u), list(u[:6]))
print("obsm:", list(a.obsm), "obsp:", list(a.obsp), "layers:", list(a.layers), "uns:", list(a.uns))
d = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
X = d(a.X[:200])
print("X min/max:", X.min(), X.max(), "integer-like:", np.allclose(X, X.round()), "var sample:", list(a.var_names[:5]))
if a.raw is not None:
    R = d(a.raw.X[:200])
    print("raw min/max:", R.min(), R.max(), "(negative = scaled, do not use)")
if "spatial" in a.obsm:
    print(
        "coords min/max:",
        a.obsm["spatial"].min(0),
        a.obsm["spatial"].max(0),
        "uns spatial:",
        list(a.uns.get("spatial", {})),
    )
```

## Selection

| Data and aim | Read |
|---|---|
| Dissociated single cells, one dataset: rank LR interactions between cell types | [single-cell-lr.md](references/single-cell-lr.md) |
| Dissociated single cells, several samples or conditions: what changes between them | [multisample.md](references/multisample.md) |
| Spatial spots (Visium) or cells: where do ligand and receptor co-vary in space, local and global scores | [spatial-bivariate.md](references/spatial-bivariate.md) |
| Spatial single cells with cell-type labels (Xenium, MERFISH, CosMx): which types signal to which via which LRs | [inflow.md](references/inflow.md) |
| Spatial single cells: at what distance do interactions occur, spatial scale of cell-type pairs | [lric.md](references/lric.md) |
| Spatial, unsupervised: what in a spot's neighbourhood predicts its expression, multi-view learning | [misty.md](references/misty.md) |
| Metabolite-mediated CCC, estimated from RNA or measured (MSI) | [metabolites.md](references/metabolites.md) |
| Resources, organism translation, custom LR lists, gene sets from LRs | [prior-knowledge.md](references/prior-knowledge.md) |
| Reading `liana_res`, score semantics, plotting | [outputs-and-plotting.md](references/outputs-and-plotting.md) |

The table is a starting point, not a rule: methods are modular and combine. Multi-modal input is
not a branch. Every method takes a `MuData`; each file ends with a Variants section saying how, and
offer those variants only when the data calls for them.

**Widen the question.** Every branch also covers non-protein mediators. Any RNA dataset, dissociated or
spatial, can be scored for metabolite-mediated CCC by estimating metabolite abundance from enzyme and
transporter expression (MetalinksDB), and a second modality (protein, measured metabolites) can
supply the ligands or the receptors. Offer this whenever the user wants the complete picture, names
metabolites, neurotransmitters, hormones or lipids, or has such a modality: read `metabolites.md`.

## Facts that apply everywhere

- **Input**: (typically) non-negative, library-size normalised, log1p expression in `.X` (or `layer=`).
  `use_raw` defaults to `False`. Scaled or z-scored values give NaN `logfc` and NaN
  `specificity_rank`, and negative values break the single-cell methods. Raw counts only trigger a warning.
  Spatial methods with `x_transform`/`y_transform` (bivariate, MISTy) also accept scaled input.
- **"Please check if appropriate organism/ID type was provided!"** means the resource and
  `var_names` do not overlap. Tell the user both causes: `var_names` that are not gene symbols
  (Ensembl IDs, the wrong matrix), and non-human data with the human `consensus` resource. For
  mouse use `resource_name="mouseconsensus"`; for a fuller map or any other organism translate
  the resource with HCOP orthologs (`li.rs.get_hcop_orthologs`, read `prior-knowledge.md`).
- **Complexes**: subunits joined by `_`. `ligand` / `receptor` columns hold the least-expressed
  subunit; `ligand_complex` / `receptor_complex` hold the full name.
- **Where results land**: single-cell methods write `adata.uns["liana_res"]` in place;
  `bivariate` and `inflow` return a **new** AnnData; `lric`, `cross_pcf` and MISTy write `.uns` keys.
- **Two thresholds**, both 0.05 by default and worth tuning: `expr_prop` (single-cell methods) is
  the fraction of cells within a cell-type group expressing a gene; `nz_prop` (spatial methods only)
  is the fraction of all cells or spots with a non-zero value.
- Plot with `li.pl.*` (plotnine, returns a `ggplot`). Do not rebuild these plots from matplotlib primitives.
- Extras: MOFA, Tensor-cell2cell, pseudobulk DE, causal networks and MetalinksDB need
  `pip install 'liana[extras]'`. Before writing code for such a route, import the package it needs
  (`pydeseq2`, `decoupler`, `muon`, `cell2cell`, `corneto`) and, on ImportError, give the user that
  command first. Downloads (`li.ds.kang_2018`, HCOP tables, MetalinksDB) go to the cwd.

## Citing

Always cite LIANA+ (Dimitrov et al., Nat Cell Biol 2024, doi:10.1038/s41556-024-01469-w) plus the
original paper of the method and resource used: `li.mt.<method>.reference` holds each single-cell
method's citation, and `li.rs.show_resources()` names the resource databases. For the consensus resource
and rank aggregate also cite Dimitrov et al., Nat Commun 2022 (doi:10.1038/s41467-022-30755-0). Inflow and
LRIC are unpublished (Alsayah et al., in preparation): cite LIANA+ for them meanwhile.
