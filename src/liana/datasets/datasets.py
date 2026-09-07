from __future__ import annotations

from typing import TYPE_CHECKING, cast

from liana._core._types import get_obs, get_x
from liana.datasets._fetch import fetch_dataset

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData


def kang_2018() -> AnnData:
    """
    Load the data from Kang et al., 2018; GSE96583.

    The data contains ~25k PBMCs cells from 8 pooled patient lupus samples, each before and after IFN-beta stimulation.
    Kang, H., Subramaniam, M., Targ, S. et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation. Nat Biotechnol 36, 89-94 (2018). https://doi.org/10.1038/nbt.4042

    The dataset was preprocessed for and is available via pertpy (https://github.com/theislab/pertpy; Heumos et al., In prep.).

    Returns
    -------
    Returns a largely pre-processed AnnData object with the following attributes:
    Raw counts for ~25k cells; ~15k genes; 16 samples; 2 conditions.

    Examples
    --------
    This downloads ~40MB into :attr:`scanpy.settings.datasetdir` on first call, so it is not run here::

        import liana as li

        adata = li.ds.kang_2018()

    The resulting object carries `obs['sample']`, `obs['condition']`, `obs['patient']` and `obs['cell_abbr']`, and is the starting point of several tutorials.
    """
    adata = cast("AnnData", fetch_dataset("kang_2018"))

    # Store the counts for later use
    adata.layers["counts"] = get_x(adata).copy()
    # Rename label to condition, replicate to patient
    adata.obs = get_obs(adata).rename({"label": "condition", "replicate": "patient"}, axis=1)

    # assign sample
    obs = get_obs(adata)
    obs["sample"] = obs["condition"].astype("str") + "&" + obs["patient"].str.slice(8, 13)

    # set cell_types abbreviations (recommended given MOFA appends names)
    abbreviations = {
        "CD4 T cells": "CD4T",
        "B cells": "B",
        "NK cells": "NK",
        "CD8 T cells": "CD8T",
        "FCGR3A+ Monocytes": "FGR3",
        "CD14+ Monocytes": "CD14",
        "Dendritic cells": "DCs",
        "Megakaryocytes": "Mega",
    }
    obs["cell_abbr"] = obs["cell_type"].astype(str).replace(abbreviations).astype("category")

    return adata


def kuppe_2022() -> AnnData:
    """
    Load a single 10X Visium slide from Kuppe et al., 2022.

    The slide (`Visium_19_CK297`) is taken from the ischemic zone of the heart of a patient with myocardial infarction.
    Kuppe, C., Ramirez Flores, R.O., Li, Z. et al. Spatial multi-omic map of human myocardial infarction. Nature 608, 766-777 (2022). https://doi.org/10.1038/s41586-022-05060-x

    Returns
    -------
    Returns an AnnData object with raw counts for ~4k spots, along with the spatial coordinates in `obsm['spatial']` and the cell type compositions of each spot in `obsm['compositions']`.

    Examples
    --------
    This downloads ~45MB into :attr:`scanpy.settings.datasetdir` on first call, so it is not run here::

        import liana as li

        adata = li.ds.kuppe_2022()

    It is the slide used in the bivariate metrics and MISTy tutorials.
    """
    return cast("AnnData", fetch_dataset("kuppe_2022"))


def citeseq_pbmc5k() -> MuData:
    """
    Load the processed 10X 5k PBMC CITE-seq data.

    The RNA and protein modalities were processed following the muon CITE-seq tutorial (https://muon-tutorials.readthedocs.io/en/latest/cite-seq/1-CITE-seq-PBMC-5k.html).

    Returns
    -------
    Returns a MuData object with a pre-processed `mod['rna']` and `mod['prot']` modality, with the cell type annotations of the 3.9k shared cells in `mod['rna'].obs['celltype']`.

    Examples
    --------
    This downloads ~80MB into :attr:`scanpy.settings.datasetdir` on first call, so it is not run here::

        import liana as li

        mdata = li.ds.citeseq_pbmc5k()
        rna, prot = mdata.mod["rna"], mdata.mod["prot"]

    It is the dataset used in the multi-modal ligand-receptor tutorial.
    """
    return cast("MuData", fetch_dataset("citeseq_pbmc5k"))


def vicari_2024() -> MuData:
    """
    Load a single spatial multimodal analysis (SMA) slide from Vicari et al., 2024.

    The slide comes from a murine Parkinson's disease model, in which one hemisphere was subjected to unilateral 6-hydroxydopamine-induced lesions while the other remained intact.
    Vicari, M., Mirzazadeh, R., Nilsson, A. et al. Spatial multimodal analysis of transcriptomes and metabolomes in tissues. Nat Biotechnol 42, 1046-1050 (2024). https://doi.org/10.1038/s41587-023-01937-y

    Returns
    -------
    Returns a MuData object with three modalities of the same slide:
    `mod['rna']` with log1p-transformed 10X Visium counts, `mod['msi']` with total-ion-count-normalised MALDI-MSI peaks, and `mod['ct']` with the Tangram cell type proportions of the RNA modality.
    The MSI observations are not aligned to the Visium spots; aligning them is part of the corresponding tutorial.

    Examples
    --------
    This downloads ~165MB into :attr:`scanpy.settings.datasetdir` on first call, so it is not run here::

        import liana as li

        mdata = li.ds.vicari_2024()
        rna, msi, ct = mdata.mod["rna"], mdata.mod["msi"], mdata.mod["ct"]
    """
    return cast("MuData", fetch_dataset("vicari_2024"))


def yao_2023() -> AnnData:
    """
    Load the `WB_MERFISH_animal2_coronal` slide of the whole mouse brain atlas from Yao et al., 2023.

    The data was generated with MERFISH, which profiles the expression of more than 1,000 genes at subcellular spatial resolution.
    Yao, Z., van Velthoven, C.T.J., Kunst, M. et al. A high-resolution transcriptomic and spatial atlas of cell types in the whole mouse brain. Nature 624, 317-332 (2023). https://doi.org/10.1038/s41586-023-06808-9

    The object is the CELLxGENE release of the dataset (https://cellxgene.cziscience.com/collections/0cca8620-8dee-45d0-aef5-23f032a5cf09).

    Returns
    -------
    Returns the AnnData object as released on CELLxGENE: ~4M cells; ~1.1k genes indexed by Ensembl ID (with symbols in `var['gene_name']`), cell type annotations in `obs`, and the spatial coordinates in `obsm['spatial']`.

    Examples
    --------
    This downloads ~1GB into :attr:`scanpy.settings.datasetdir` on first call, so it is not run here::

        import liana as li

        adata = li.ds.yao_2023()

    It is the dataset used in the inflow and LRIC tutorials.
    """
    return cast("AnnData", fetch_dataset("yao_2023"))
