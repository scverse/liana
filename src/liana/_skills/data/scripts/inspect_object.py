"""Summarise an AnnData/MuData object for the liana skill's intake step.

Usage: ``python inspect_object.py <file.h5ad | file.h5mu | Space Ranger outs/ folder>``

Prints what liana needs to know: modalities, candidate label columns, which matrix holds
non-negative log-normalised expression, the organism and ID type of ``var_names``, and spatial
coordinates with their unit scale factors. Imports neither liana nor anything beyond the
AnnData stack, so it runs wherever the data can be read.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Rows read per matrix: enough to judge sign, integrality and range without densifying a large object.
N_ROWS = 200
# Unique values listed per categorical column; more only clutters the output.
N_UNIQUE = 8
# Columns with more distinct values than this are identifiers (barcodes), not labels.
MAX_LEVELS = 200


def _read_spaceranger(outs: Path) -> Any:
    """Counts, coordinates and scale factors of a Space Ranger ``outs/`` folder (images skipped)."""
    import json

    import pandas as pd
    import scanpy as sc

    a = sc.read_10x_h5(outs / "filtered_feature_bc_matrix.h5")
    a.var_names_make_unique()
    spatial = outs / "spatial"
    pos_file = next(
        (p for p in (spatial / "tissue_positions.csv", spatial / "tissue_positions_list.csv") if p.exists()), None
    )
    if pos_file is not None:
        header = 0 if pos_file.name == "tissue_positions.csv" else None
        pos = pd.read_csv(pos_file, header=header, index_col=0)
        pos = pos.reindex(a.obs_names)
        a.obsm["spatial"] = pos.iloc[:, [4, 3]].to_numpy(dtype=float)  # pxl_col (x), pxl_row (y)
    sf_file = spatial / "scalefactors_json.json"
    if sf_file.exists():
        a.uns["spatial"] = {outs.parent.name or "library": {"scalefactors": json.loads(sf_file.read_text())}}
    return a


def _read(path: Path) -> Any:
    if path.is_dir():
        return _read_spaceranger(path)
    if path.suffix == ".h5mu":
        import mudata as mu

        return mu.read_h5mu(path)
    import anndata as ad

    return ad.read_h5ad(path)


def _dense(m: Any) -> np.ndarray:
    m = m[:N_ROWS]
    return np.asarray(m.toarray() if hasattr(m, "toarray") else m, dtype=float)


def _range(name: str, m: Any) -> str:
    x = _dense(m)
    if x.size == 0:
        return f"{name}: empty"
    integral = bool(np.allclose(x, np.round(x)))
    note = ""
    if x.min() < 0:
        note = "  <- negative = scaled; not for single-cell LR methods"
    elif integral and x.max() > 1:
        note = "  <- integer-like = raw counts; normalise + log1p first"
    elif x.max() > 50:
        note = "  <- max > 50: not log1p-normalised"
    return f"{name}: min {x.min():.3g} max {x.max():.3g} integer-like {integral}{note}"


def _organism(var_names: Any) -> str:
    names = [str(v) for v in list(var_names)[:2000]]
    if not names:
        return "no var_names"
    if sum(n.startswith(("ENSG", "ENSMUSG", "ENS")) for n in names) > len(names) / 2:
        return "Ensembl IDs: map to gene symbols before using an LR resource"
    letters = [n for n in names if n[:1].isalpha() and not n.startswith(("MT-", "mt-"))]
    upper = sum(n.isupper() for n in letters)
    title = sum(n[:1].isupper() and n[1:] == n[1:].lower() and n[1:].isalpha() for n in letters)
    if upper > 2 * title:
        return "symbols, upper case (CD4): human-like"
    if title > 2 * upper:
        return "symbols, title case (Cd4): mouse-like -> mouseconsensus or HCOP translation"
    return "unclear symbol case: ask the species"


def _describe_modality(name: str, m: Any) -> None:
    prefix = f"[{name}] " if name else ""
    print(f"{prefix}shape {m.shape}")
    print(f"{prefix}{_range('X', m.X)}" if m.X is not None else f"{prefix}X: None")
    for key in m.layers:
        if key is None:  # anndata >= 0.13 lists .X as the `None` layer
            continue
        print(f"{prefix}{_range(f'layer {key!r}', m.layers[key])}")
    if m.raw is not None:
        print(f"{prefix}{_range('.raw', m.raw.X)}")
    print(f"{prefix}var_names sample: {list(m.var_names[:8])}")
    print(f"{prefix}var_names: {_organism(m.var_names)}")


def _describe_obs(obs: Any, label: str = "obs") -> None:
    for col in obs.columns:
        s = obs[col]
        if s.dtype.kind in "biuf":
            continue
        levels = s.astype("string").unique()
        if 1 < len(levels) <= MAX_LEVELS:
            shown = [str(v) for v in levels[:N_UNIQUE]]
            print(f"{label}[{col!r}]: {len(levels)} levels {shown}")


def _describe_spatial(a: Any) -> None:
    if "spatial" not in a.obsm:
        print("spatial: no obsm['spatial']")
        return
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    print(f"spatial coords min {xy.min(0).round(1).tolist()} max {xy.max(0).round(1).tolist()}")
    libs = a.uns.get("spatial", {})
    if not libs:
        print("spatial: no uns['spatial'] scalefactors; ask the coordinate unit (µm or pixels)")
    for lib, entry in libs.items():
        sf = dict(entry.get("scalefactors", {}))
        print(f"uns['spatial'][{lib!r}] scalefactors: {sf}")
        # a Visium spot is 55 µm wide; full-resolution coordinates are pixels
        um_per_px = sf.get("microns_per_pixel") or (
            55 / sf["spot_diameter_fullres"] if "spot_diameter_fullres" in sf else None
        )
        if um_per_px:
            print(f"  -> µm per full-res pixel ~ {float(um_per_px):.4g}; divide µm distances by it for pixel units")


def main(argv: list[str]) -> int:
    """Print the summary for ``argv[0]``; return a shell exit code."""
    if len(argv) != 1:
        print(__doc__.splitlines()[2], file=sys.stderr)
        return 2
    path = Path(argv[0]).expanduser()
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1
    try:
        a = _read(path)
    except Exception as e:  # noqa: BLE001 - report any reader failure rather than a traceback
        print(f"error: could not read {path}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    mods = dict(getattr(a, "mod", {}))
    print(f"type {type(a).__name__} shape {a.shape} modalities {list(mods)}")
    _describe_obs(a.obs)
    print(f"obsm {list(a.obsm)} obsp {list(a.obsp)} uns {list(a.uns)}")
    if mods:  # a MuData's container has no usable .X: describe each modality
        for name, m in mods.items():
            _describe_obs(m.obs, f"[{name}] obs")  # labels often live only on a modality
            print(f"[{name}] obsm {list(m.obsm)} obsp {list(m.obsp)}")
            _describe_modality(name, m)
    else:
        _describe_modality("", a)
    _describe_spatial(a)
    for name, m in mods.items():
        if "spatial" in m.obsm and "spatial" not in a.obsm:
            print(f"[{name}] ", end="")
            _describe_spatial(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
