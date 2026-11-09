"""Execute the skill's ``python`` code blocks, in file order, on toy data.

Each file's blocks share one namespace, preloaded with the modules the skill says snippets assume
and with toy objects bound to the names the blocks use. A block preceded by
``<!-- skip-exec: network -->`` runs only in that file's ``network``-marked variant;
``<!-- skip-exec: extras -->`` blocks (MOFA, pydeseq2, Tensor-cell2cell, corneto: slow) are skipped
unless ``LIANA_SKILL_EXEC_EXTRAS=1``.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import mudata as mu
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import liana as li
from liana._skills.install import bundled_skill_dir
from tests._helpers import as_frame, get_csr

DATA = bundled_skill_dir()
ALL_MD = [DATA / "SKILL.md", *sorted((DATA / "references").glob("*.md"))]
BLOCK = re.compile(r"(?:<!-- skip-exec: (\w+) -->\n)?```python\n(.*?)```", re.S)
RUN_EXTRAS = os.environ.get("LIANA_SKILL_EXEC_EXTRAS") == "1"
CACHE = Path(__file__).parents[1] / ".cache"  # shared with tests/resource's downloads


def _blocks(md: Path) -> list[tuple[str | None, str]]:
    return [(m.group(1), m.group(2)) for m in BLOCK.finditer(md.read_text())]


# --- toy objects ------------------------------------------------------------------------------


def _adata() -> ad.AnnData:
    adata = li.ds.generate_toy_adata()
    obs = as_frame(adata.obs)
    obs["cell_type"] = obs["bulk_labels"]
    obs["condition"] = np.where(obs["sample"].isin(["A", "B"]), "stim", "ctrl")
    adata.layers["counts"] = np.round(np.expm1(get_csr(adata).toarray()))
    return adata


def _spatial() -> ad.AnnData:
    """Toy spatial data on a µm-like scale: 700 cells over ~250 µm, so ~10 µm apart."""
    adata = li.ds.generate_toy_spatial()
    obs = as_frame(adata.obs)
    obs["cell_type"] = obs["bulk_labels"]
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"]) * 0.05
    return adata


def _ns_single_cell() -> dict[str, Any]:
    adata = _adata()
    prot = adata[:, :200].copy()
    return {"adata": adata, "mdata": mu.MuData({"rna": adata.copy(), "prot": prot})}


def _ns_outputs() -> dict[str, Any]:
    adata = _adata()
    li.mt.rank_aggregate(adata, groupby="cell_type", n_perms=None)
    labels = adata.obs["cell_type"].cat.categories
    return {"adata": adata, "sources": list(labels[:3]), "targets": list(labels[3:6])}


def _ns_multisample() -> dict[str, Any]:
    ns = _ns_outputs()
    ns["adata"] = _adata()
    ppis = [("CD4", 1, "LCK"), ("LCK", 1, "JUN"), ("LCK", -1, "FOS")]
    ns.update(
        ppis=ppis,
        receptor_scores={"CD4": 1.0},
        tf_scores={"JUN": 1.0, "FOS": -1.0},
        expr_props={"CD4": 0.5, "LCK": 0.5, "JUN": 0.5, "FOS": 0.5},
        sources=ns["sources"][0],
        targets=ns["targets"][0],
    )
    return ns


def _ns_spatial() -> dict[str, Any]:
    adata = _spatial()
    return {"adata": adata, "receivers": list(adata.obs["cell_type"].cat.categories[:2])}


def _ns_spatial_setup() -> dict[str, Any]:
    adata = li.ds.generate_toy_spatial()
    # full-res pixels: 55 µm spot / 1100 px = 0.05 µm per pixel, as in `_spatial`
    adata.uns["spatial"] = {"lib": {"scalefactors": {"spot_diameter_fullres": 1100.0}}}
    return {"adata": adata}


def _ns_bivariate() -> dict[str, Any]:
    mdata = li.ds.generate_toy_mdata()
    mdata.obsm["spatial"] = mdata.obsm["spatial"] * 0.05
    li.pp.spatial_neighbors(mdata, bandwidth=100)
    # x != y: identical names would be dropped as self-interactions
    interactions = list(zip(mdata.mod["adata_x"].var_names[:-1], mdata.mod["adata_y"].var_names[1:], strict=True))
    return {**_ns_spatial(), "mdata": mdata, "interactions": interactions}


# enzymes and receptors of a few MetalinksDB metabolites, absent from the toy's variable genes
METABOLITE_GENES = ["AOC1", "AOC2", "AOC3", "DDC", "HDC", "HNMT", "COMT", "DBH", "MAOA", "MAOB", "AANAT", "INMT"]
METABOLITE_GENES += ["PTGES", "PTGES2", "PTGES3", "CBR1", "AKR1C3", "ADORA1", "ADORA2A", "ACKR3"]
METABOLITE_GENES += ["SLC18A1", "SLC18A2", "SLC22A1", "SLC22A2", "SLC22A3", "SLC6A4", "SLC29A4"]  # transporters


def _ns_metabolites() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    toy = _adata()
    extra = ad.AnnData(
        np.log1p(rng.poisson(0.5, size=(toy.n_obs, len(METABOLITE_GENES))).astype(float)),
        obs=pd.DataFrame(index=toy.obs_names),
        var=pd.DataFrame(index=METABOLITE_GENES),
    )
    adata = ad.concat([toy, extra], axis=1, merge="first")
    adata.obs = as_frame(toy.obs)
    rna = _spatial()
    metabolites = ["Histamine", "Dopamine", "Serotonin", "Glycerol"]
    msi = ad.AnnData(
        rng.gamma(1.0, size=(300, len(metabolites))),
        obs=pd.DataFrame(index=[f"px{i}" for i in range(300)]),
        var=pd.DataFrame(index=metabolites),
    )
    xy = np.asarray(rna.obsm["spatial"])
    msi.obsm["spatial"] = rng.uniform(xy.min(0), xy.max(0), size=(300, 2))
    return {"adata": adata, "rna": rna, "msi": msi}


FIXTURES: dict[str, Callable[[], dict[str, Any]]] = {
    "SKILL.md": dict,
    "single-cell-lr.md": _ns_single_cell,
    "outputs-and-plotting.md": _ns_outputs,
    "multisample.md": _ns_multisample,
    "prior-knowledge.md": lambda: {"adata": _adata()},
    "spatial-setup.md": _ns_spatial_setup,
    "spatial-bivariate.md": _ns_bivariate,
    "inflow.md": _ns_spatial,
    "lric.md": _ns_spatial,
    "misty.md": _ns_spatial,
    "metabolites.md": _ns_metabolites,
    "troubleshooting.md": dict,
}


# results the blocks must leave non-empty: a call that runs but scores nothing is a broken example
CHECKS: dict[str, list[str]] = {
    "single-cell-lr.md": ["len(adata.uns['liana_res'])", "len(mdata.uns['liana_res'])"],
    "outputs-and-plotting.md": ["len(adata.uns['liana_res'])"],
    "multisample.md": ["adata.uns['liana_res']['sample'].nunique() == 4"],
    "spatial-setup.md": ["adata.obsp['spatial_connectivities'].nnz > adata.n_obs"],
    "spatial-bivariate.md": ["lrdata.n_vars"],
    "inflow.md": ["lrdata.n_vars", "len(lrdata.uns['global_interactions'])"],
    "lric.md": ["adata.uns['lric']['g'].notna().sum()"],
    "misty.md": ["len(misty.uns['target_metrics'])", "len(misty.uns['interactions'])"],
    "metabolites.md": ["len(meta.uns['liana_res'])", "lrdata.n_vars"],
}


# --- tests -----------------------------------------------------------------------------------


def test_every_file_has_a_fixture() -> None:
    assert {md.name for md in ALL_MD} == set(FIXTURES)


@pytest.mark.parametrize("md", ALL_MD, ids=lambda p: p.name)
def test_blocks_are_complete(md: Path) -> None:
    for marker, block in _blocks(md):
        assert marker in (None, "network", "extras"), f"{md.name}: unknown skip-exec marker {marker!r}"
        assert "..." not in block, f"{md.name}: elided code in\n{block}"


def _params() -> list[Any]:
    params = []
    for md in ALL_MD:
        markers = {m for m, _ in _blocks(md)}
        params.append(pytest.param(md, False, id=md.name))
        if "network" in markers:
            params.append(pytest.param(md, True, id=f"{md.name}-network", marks=pytest.mark.network))
    return params


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(("md", "network"), _params())
def test_blocks_run(md: Path, network: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocks = [b for m, b in _blocks(md) if m is None or (m == "network" and network) or (m == "extras" and RUN_EXTRAS)]
    if not blocks:
        pytest.skip("no executable blocks")
    monkeypatch.chdir(tmp_path)  # snippets save plots and models to the working directory
    datasetdir = sc.settings.datasetdir
    if network:
        CACHE.mkdir(exist_ok=True)
        sc.settings.datasetdir = CACHE
    ns: dict[str, Any] = {"li": li, "sc": sc, "np": np, "pd": pd, "mu": mu, "ad": ad, **FIXTURES[md.name]()}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, block in enumerate(blocks):
                exec(compile(block, f"{md.name}[block {i}]", "exec"), ns)
        for check in CHECKS.get(md.name, []):
            assert eval(check, ns), f"{md.name}: `{check}` is empty"
    finally:
        sc.settings.datasetdir = datasetdir
        plt.close("all")
