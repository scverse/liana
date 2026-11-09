"""Run the skill's intake script, ``scripts/inspect_object.py``, as an agent would: in a subprocess."""

import json
import subprocess
import sys
from pathlib import Path

import mudata as mu
import numpy as np
import pytest

from liana._skills.install import bundled_skill_dir
from liana.datasets import generate_toy_adata, generate_toy_mdata, generate_toy_spatial
from tests._helpers import as_frame, get_csr

SCRIPT = bundled_skill_dir() / "scripts" / "inspect_object.py"


def _run(path: Path) -> str:
    proc = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_h5ad(tmp_path: Path) -> None:
    adata = generate_toy_adata()
    adata.layers["counts"] = np.round(np.expm1(get_csr(adata).toarray()))
    path = tmp_path / "toy.h5ad"
    adata.write_h5ad(path)
    out = _run(path)
    assert "type AnnData shape (700, 765)" in out
    assert "obs['bulk_labels']: 10 levels" in out
    assert "obs['sample']: 4 levels" in out
    assert "layer 'counts'" in out and "raw counts" in out
    assert ".raw: min 0" in out
    assert "human-like" in out
    assert "no obsm['spatial']" in out


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_h5mu(tmp_path: Path) -> None:
    path = tmp_path / "toy.h5mu"
    generate_toy_mdata().write_h5mu(path)
    out = _run(path)
    assert "type MuData" in out and "modalities ['adata_x', 'adata_y']" in out
    assert "[adata_x] X: min 0" in out
    assert "[adata_x] layer 'scaled'" in out and "negative = scaled" in out
    assert "spatial coords min" in out


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_h5mu_labels_on_a_modality(tmp_path: Path) -> None:
    rna = generate_toy_adata()
    prot = rna[:, :20].copy()
    prot.obs = as_frame(prot.obs).drop(columns="bulk_labels")
    path = tmp_path / "cite.h5mu"
    mu.MuData({"rna": rna, "prot": prot}).write_h5mu(path)
    out = _run(path)
    assert "[rna] obs['bulk_labels']: 10 levels" in out
    assert "[prot] obs['bulk_labels']" not in out


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_spatial_scalefactors_and_mouse(tmp_path: Path) -> None:
    adata = generate_toy_spatial()
    adata.var_names = [v.capitalize() for v in adata.var_names]
    adata.uns["spatial"] = {"lib": {"scalefactors": {"spot_diameter_fullres": 110.0}}}
    path = tmp_path / "spatial.h5ad"
    adata.write_h5ad(path)
    out = _run(path)
    assert "mouse-like" in out
    assert "spot_diameter_fullres" in out
    assert "µm per full-res pixel ~ 0.5" in out


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_ensembl_ids(tmp_path: Path) -> None:
    adata = generate_toy_adata()
    adata.var_names = [f"ENSG{i:011d}" for i in range(adata.n_vars)]
    del adata.raw
    path = tmp_path / "ens.h5ad"
    adata.write_h5ad(path)
    assert "Ensembl IDs" in _run(path)


def test_spaceranger_outs(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    outs = tmp_path / "outs"
    (outs / "spatial").mkdir(parents=True)
    barcodes = [f"AAAC{i}-1" for i in range(6)]
    genes = ["Cd4", "Cxcl9", "Apoe", "Lrp1"]
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=(len(genes), len(barcodes)))  # 10x stores features x barcodes, CSC
    data: list[int] = []
    indices: list[int] = []
    indptr = [0]
    for j in range(counts.shape[1]):
        nz = np.flatnonzero(counts[:, j])
        data.extend(counts[nz, j].tolist())
        indices.extend(nz.tolist())
        indptr.append(len(data))
    with h5py.File(outs / "filtered_feature_bc_matrix.h5", "w") as f:
        f.attrs["library_ids"] = np.array(["lib"], dtype="S")
        g = f.create_group("matrix")
        g["barcodes"] = np.array(barcodes, dtype="S")
        g["data"] = np.array(data, dtype="int32")
        g["indices"] = np.array(indices, dtype="int64")
        g["indptr"] = np.array(indptr, dtype="int64")
        g["shape"] = np.array(counts.shape, dtype="int32")
        ft = g.create_group("features")
        ft["id"] = np.array([f"ENSMUSG{i}" for i in range(len(genes))], dtype="S")
        ft["name"] = np.array(genes, dtype="S")
        ft["feature_type"] = np.array(["Gene Expression"] * len(genes), dtype="S")
        ft["genome"] = np.array(["mm10"] * len(genes), dtype="S")
        ft["_all_tag_keys"] = np.array(["genome"], dtype="S")
    rows = [f"{b},1,{i},{i},{1000 + 100 * i},{2000 + 50 * i}" for i, b in enumerate(barcodes)]
    (outs / "spatial" / "tissue_positions_list.csv").write_text("\n".join(rows) + "\n")
    (outs / "spatial" / "scalefactors_json.json").write_text(json.dumps({"spot_diameter_fullres": 55.0}))
    out = _run(outs)
    assert "shape (6, 4)" in out
    assert "raw counts" in out
    assert "mouse-like" in out
    assert "spatial coords min [2000.0, 1000.0] max [2250.0, 1500.0]" in out
    assert "µm per full-res pixel ~ 1" in out


def test_bad_path(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "missing.h5ad")], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "does not exist" in proc.stderr
