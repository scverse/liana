---
name: liana
description: Infers cell-cell communication (CCC) and ligand-receptor interactions with the liana Python package (LIANA+, scverse) on AnnData/MuData objects. Covers dissociated single-cell data, multi-sample and multi-condition designs, spatial data (Visium, Xenium, MERFISH, CosMx), multimodal CITE-seq, and metabolite-mediated signalling. Includes CellPhoneDB, CellChat, rank aggregation, MISTy, Tensor-cell2cell and MOFA+. Use it whenever the user asks about cell-cell interactions, crosstalk, sender and receiver cell types, ligand-receptor pairs or niche signalling, even if liana is not named.
---

# liana (LIANA+)

`import liana as li`: `li.mt` methods, `li.pp` preprocessing, `li.rs` prior knowledge, `li.pl` plots,
`li.ms` multi-sample helpers, `li.ds` datasets. Snippets assume `scanpy as sc`, `numpy as np`,
`pandas as pd`, `mudata as mu`, `anndata as ad`.

## Workflow

1. **Intake.** Ask for the object or its path; say a description is fine if the data cannot be shared.
   With a path, run `python <skill dir>/scripts/inspect_object.py <path>` (skill dir: where this file
   lives, e.g. `~/.claude/skills/liana`; takes `.h5ad`, `.h5mu` or a Space Ranger `outs/`). From its output infer:
   - which matrix is non-negative log1p (`.X`, a layer); counts or negative values need fixing first;
   - the cell-type, sample and condition columns; modalities; organism and ID type;
   - for spatial data, the coordinate unit (scalefactors).
   Ask only what is still missing, usually the aim and which column is which. Without a file, ask
   briefly: tissue and technology, samples and grouping, annotation, species, extra modalities, aim. To analyse an `outs/` folder use `squidpy.read.visium`; Xenium,
   CosMx, MERSCOPE load with `spatialdata_io`.
2. **Select** a branch from the table.
3. **Read** that file, plus [spatial-setup.md](references/spatial-setup.md) for any spatial task,
   [prior-knowledge.md](references/prior-knowledge.md) for non-human data or custom LR lists, and
   [outputs-and-plotting.md](references/outputs-and-plotting.md) before plotting or interpreting `liana_res`.
   When liana raises an error or warning, read [troubleshooting.md](references/troubleshooting.md).
4. **Run**, plot the top hits with `li.pl.*`, list the top few as sender → receiver: ligand → receptor, then explain in one paragraph without column names: what was scored, which method and
   why, what a high score means, the main caveat. Name any bandwidth (in µm and in coordinate units)
   or threshold you chose, and end with the citations (see Citing).

| Data and aim | Read |
|---|---|
| Dissociated cells, one dataset: rank LR interactions between cell types | [single-cell-lr.md](references/single-cell-lr.md) |
| Dissociated cells, several samples or conditions: what changes | [multisample.md](references/multisample.md) |
| Spatial spots or cells: where ligand and receptor co-vary, local and global | [spatial-bivariate.md](references/spatial-bivariate.md) |
| Spatial cells with labels (Xenium, MERFISH, CosMx): which types signal to which | [inflow.md](references/inflow.md) |
| Spatial cells: at what distance interactions occur | [lric.md](references/lric.md) |
| Spatial, unsupervised: what in the neighbourhood predicts expression | [misty.md](references/misty.md) |
| Metabolite-mediated CCC, estimated from RNA or measured (MSI) | [metabolites.md](references/metabolites.md) |

Methods combine; the table is a starting point. Offer metabolite-mediated CCC (estimated from any RNA
data) when the user wants the complete picture or mentions metabolites, hormones, neurotransmitters or lipids.

## Essentials

- **Input**: non-negative, normalised log1p expression in `.X` or `layer=`; `use_raw=False` by default.
  Scaled input only for spatial methods with `x_transform`/`y_transform` (`bivariate`, `inflow`).
- **MuData**: `x_mod`/`y_mod` pick the ligand/receptor modality (in `mdata_kwargs` for single-cell methods). `lric`, `cross_pcf` and `estimate_metalinks` take AnnData only.
- **Results**: single-cell methods write `uns["liana_res"]`; `bivariate`, `inflow` return a new AnnData;
  `lric`, `cross_pcf` and MISTy write `uns` keys.
- **Thresholds**: `expr_prop` = fraction of cells within a group expressing a gene (single-cell methods,
  directed `lric`); `nz_prop` = fraction of all cells/spots non-zero (spatial methods); MISTy uses
  `nz_threshold` (0.1). Both props default to 0.05; tune them.
- **Complexes**: subunits joined by `_`; `ligand`/`receptor` hold the least-expressed subunit. CellChat
  picks it by trimean and reports `*_trimean` and `mat_max` instead of `*_means`.
- **Plots**: use `li.pl.*`; they return a plotnine `ggplot`, except `circle` (Axes), `annulus` (Figure),
  `feature_by_group` ((Figure, Axes)).
- **Extras** (MOFA, Tensor-cell2cell, pseudobulk DE, causal networks, metabolite estimation, NMF):
  import the package first (`muon`, `cell2cell`, `pydeseq2`, `corneto`, `decoupler`, `kneed`); on
  ImportError give `pip install 'liana[extras]'`.
- Downloads (datasets, HCOP, MetalinksDB) cache under `sc.settings.datasetdir` (default `./data`).

## Citing

Cite LIANA+ (Dimitrov et al., Nat Cell Biol 2024, doi:10.1038/s41556-024-01469-w) plus the method and
resource used: `li.mt.<method>.reference`, `li.rs.show_resources()`. For the consensus resource and
rank aggregate add Dimitrov et al., Nat Commun 2022 (doi:10.1038/s41467-022-30755-0); for MetalinksDB, Farr
et al., Brief Bioinform 2024 (doi:10.1093/bib/bbae347). Inflow and LRIC are
unpublished (Alsayah et al., in preparation): cite LIANA+ for them.
