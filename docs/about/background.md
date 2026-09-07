# About LIANA+

LIANA+ scores cell-cell interactions in single-cell, spatially-resolved and multi-modal data.
The methods it covers were published as separate tools that take different inputs and return different outputs; LIANA+ reimplements them behind one interface {cite:p}`Dimitrov_2024`, on {class}`~anndata.AnnData` {cite:p}`Virshup_2024` and {class}`~mudata.MuData` {cite:p}`Bredikhin_2022`.
If you use it in your work, see {doc}`cite`.

## Design

### One interface, many methods

The methods often disagree, and which one is appropriate depends on the data and the question {cite:p}`Dimitrov_2022`.
LIANA+ reimplements the scoring functions of CellPhoneDB, CellChat, Connectome, NATMI, SingleCellSignalR and scSeqComm, along with a log-fold-change score and a geometric mean, so that they take the same input, use the same permutation scheme and return the same table.
That makes the scores comparable, and {meth}`li.mt.rank_aggregate <liana.mt.rank_aggregate.__call__>` combines them into a rank consensus.

### Methods and prior knowledge are separate

Methods score an interchangeable resource.
{func}`li.rs.select_resource <liana.rs.select_resource>` returns the curated consensus resource or any of the ligand-receptor databases behind it, {func}`li.rs.translate_resource <liana.rs.translate_resource>` moves a resource across organisms, and {func}`li.rs.get_metalinks <liana.rs.get_metalinks>` replaces the ligands with metabolites.
A new resource therefore works with every method, and a new method with every resource.

### More than one dissociated dataset

The same scoring machinery covers three further settings.
In spatially-resolved data, interactions can be restricted to spatial neighborhoods, scored per spot or cell with the bivariate metrics, or modelled as spatial relationships with {class}`li.mt.MistyData <liana.mt.MistyData>` {cite:p}`Tanevski_2022` and {meth}`li.mt.lric <liana.mt.lric.__call__>`.
In multi-modal data, the ligand and the receptor can come from different modalities, such as transcriptome and surface protein, or transcriptome and MALDI-MSI metabolite.
Across samples or conditions, interaction scores can be reshaped into views and factorized with MOFA {cite:p}`Argelaguet_2020` or tensor-cell2cell {cite:p}`Armingol_2022`, which summarises a collection of samples as a few communication programs.

### Downstream of the receptor

A ligand-receptor hit on its own says little about the mechanism behind it.
{func}`li.mt.find_causalnet <liana.mt.find_causalnet>` links the receptors to the transcription factors that respond to them, by solving for the intracellular signaling network that best explains the measured activities.

## Ecosystem

LIANA+ is built on [anndata](https://anndata.readthedocs.io/) {cite:p}`Virshup_2024` and [mudata](https://mudata.readthedocs.io/) {cite:p}`Bredikhin_2022`, and is used together with [scanpy](https://scanpy.readthedocs.io/) {cite:p}`Wolf_2018`, [squidpy](https://squidpy.readthedocs.io/) {cite:p}`Palla_2022`, [decoupler](https://decoupler.readthedocs.io/) {cite:p}`Badia_i_Mompel_2022`, [omnipath](https://omnipathdb.org/) {cite:p}`T_rei_2021`, [MOFA](https://biofam.github.io/MOFA2/) {cite:p}`Argelaguet_2020`, [tensor-cell2cell](https://earmingol.github.io/cell2cell/) {cite:p}`Armingol_2022` and [corneto](https://saezlab.github.io/corneto/).
The [Saez-Rodriguez group](https://saezlab.org/) develops it, and it is part of the [scverse ecosystem](https://scverse.org/) {cite:p}`Virshup_2023`.

## Why LIANA+?

LIANA started in R as a LIgand-receptor ANalysis frAmework that benchmarked and combined the methods and resources that existed at the time.
The `+` is the Python framework that grew out of it, with the same method-agnostic core extended to spatial, multi-modal, multi-sample and intracellular analyses.
A liana is also a climbing vine, which is what the logo shows.
