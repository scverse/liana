from typing import TypedDict

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from tests._helpers import get_obs

from liana.datasets import generate_toy_spatial
from liana.datasets._sample_resource import sample_resource
from liana.method import get_lric_auc, get_lric_divergence
from liana.method.sp._LRIC import cross_pcf, lric


class _RadiusKwargs(TypedDict):
    """The radius-grid arguments the `cross_pcf`/`lric` calls here share."""

    max_radius: float
    radius_step: float
    verbose: bool


_KWARGS = _RadiusKwargs(max_radius=100, radius_step=20, verbose=False)

_ID_COLS = {
    "cross_pcf": ["source", "target", "interaction"],
    "lric_ag": ["ligand_complex", "receptor_complex", "interaction"],
    "lric_ct": ["source", "target", "ligand_complex", "receptor_complex", "interaction"],
}


@pytest.fixture(scope="module")
def adata() -> AnnData:
    ad = generate_toy_spatial()
    get_obs(ad)["cell_type"] = get_obs(ad)["bulk_labels"]
    res = sample_resource(ad, n_lrs=5, seed=42)
    cross_pcf(ad, groupby="cell_type", key_added="cross_pcf", **_KWARGS)
    lric(ad, resource=res, key_added="lric_ag", **_KWARGS)
    lric(ad, resource=res, groupby="cell_type", key_added="lric_ct", **_KWARGS)
    return ad


@pytest.mark.parametrize("uns_key", ["cross_pcf", "lric_ag", "lric_ct"])
def test_get_lric_auc(adata: AnnData, uns_key: str) -> None:
    df = get_lric_auc(adata, uns_key, max_dist=60, min_bins=2)
    # id columns match liana's convention, so the result feeds `li.pl.dotplot` directly
    assert list(df.columns) == [*_ID_COLS[uns_key], "score", "peak_radius"]
    assert len(df) > 0
    # sorted most-enriched first
    assert df["score"].is_monotonic_decreasing
    # the peak lies on the radius grid, inside the integration window
    radii = set(adata.uns[uns_key]["radius"].unique())
    assert set(df["peak_radius"]) <= {r for r in radii if r < 60}


def test_get_lric_auc_liana_res(adata: AnnData) -> None:
    """`liana_res=` is equivalent to reading the same frame off `.uns`."""
    df = get_lric_auc(liana_res=adata.uns["lric_ag"], max_dist=60, min_bins=2)
    assert df.equals(get_lric_auc(adata, "lric_ag", max_dist=60, min_bins=2))

    with pytest.raises(ValueError, match="must be provided"):
        get_lric_auc()


def test_get_lric_auc_stacked_results_raise(adata: AnnData) -> None:
    base = adata.uns["lric_ag"]
    res = pd.concat([base.assign(sample="a"), base.assign(sample="b")], ignore_index=True)

    with pytest.raises(ValueError, match=r"\['sample'\] take more than one value"):
        get_lric_auc(liana_res=res, min_bins=2)
    # one sample at a time is unambiguous and scores as before
    one = get_lric_auc(liana_res=res[res["sample"] == "a"].drop(columns="sample"), min_bins=2)
    assert one.equals(get_lric_auc(liana_res=base, min_bins=2))

    # two samples on DISJOINT radius grids share no (ids, radius) key at all, so a
    # duplicate-key guard waved them through and their curves interleaved into one row
    curve = pd.DataFrame({"interaction": ["a^b"] * 3, "g": [0.5, 1.0, 2.0]})
    disjoint = pd.concat(
        [
            curve.assign(radius=[10.0, 20.0, 30.0], condition="ctrl"),
            curve.assign(radius=[15.0, 25.0, 35.0], condition="stim"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match=r"\['condition'\] take more than one value"):
        get_lric_auc(liana_res=disjoint, min_bins=2)


def test_min_bins_gates_out_everything(adata: AnnData) -> None:
    # more bins required than exist in the window -> empty, but well-formed
    df = get_lric_auc(adata, "cross_pcf", max_dist=25, min_bins=99)
    assert list(df.columns) == [*_ID_COLS["cross_pcf"], "score", "peak_radius"]
    assert df.empty


def test_get_lric_divergence(adata: AnnData) -> None:
    two = adata.uns["cross_pcf"]["interaction"].unique()[:2]
    div = get_lric_divergence(
        adata,
        "cross_pcf",
        feature_a={"interaction": two[0]},
        feature_b={"interaction": two[1]},
        min_bins=2,
    )
    assert div["divergence"] > 0
    assert div["r_star"] in set(adata.uns["cross_pcf"]["radius"])
    assert div["direction"] == ("A > B" if div["delta_star"] > 0 else "B > A")

    # a curve against itself is exactly zero divergence
    self_div = get_lric_divergence(
        adata,
        "cross_pcf",
        feature_a={"interaction": two[0]},
        feature_b={"interaction": two[0]},
        min_bins=2,
    )
    assert self_div["divergence"] == 0.0
    assert self_div["direction"] == "equal"


def test_get_lric_divergence_across_conditions(adata: AnnData) -> None:
    # concatenated results with a `condition` column: pin it in the selections
    # to compare the same interaction across conditions
    base = adata.uns["lric_ag"]
    stim = base.assign(condition="stim", g=base["g"] * 2)  # exactly +1 in log2
    res = pd.concat([base.assign(condition="ctrl"), stim], ignore_index=True)

    lr = base["interaction"].dropna().unique()[0]
    # `np.log2` pins the strict math -- the default floors g at 0.05, which
    # would compress the exact +1 log2 shift for bins with small g
    div = get_lric_divergence(
        liana_res=res,
        feature_a={"interaction": lr, "condition": "stim"},
        feature_b={"interaction": lr, "condition": "ctrl"},
        min_bins=2,
        transform_fn=np.log2,
    )
    assert div["divergence"] == pytest.approx(1.0)
    assert div["delta_star"] == pytest.approx(1.0)

    # without pinning `condition`, the two replicates average into one curve
    avg = get_lric_divergence(
        liana_res=res,
        feature_a={"interaction": lr},
        feature_b={"interaction": lr},
        min_bins=2,
    )
    assert avg["divergence"] == 0.0


def test_floored_default_keeps_zero_g_bins(adata: AnnData) -> None:
    # the default transform floors g at 0.05, so a g=0 bin stays finite;
    # strict np.log2 drops it (-inf) and the interaction falls below min_bins
    res = adata.uns["lric_ag"].copy()
    lr = res["interaction"].iloc[0]
    res.loc[res.index[res["interaction"] == lr][0], "g"] = 0.0
    n_bins = res.loc[res["interaction"] == lr, "radius"].nunique()

    floored = get_lric_auc(liana_res=res, min_bins=n_bins)
    assert lr in set(floored["interaction"])
    strict = get_lric_auc(liana_res=res, min_bins=n_bins, transform_fn=np.log2)
    assert lr not in set(strict["interaction"])


def test_get_lric_divergence_errors(adata: AnnData) -> None:
    with pytest.raises(ValueError, match="must be provided"):
        get_lric_divergence(adata, "cross_pcf", feature_a={"interaction": "x"})
    # ambiguous selection: many pairs share a source
    src = adata.uns["cross_pcf"]["source"].unique()[0]
    with pytest.raises(ValueError, match="more than one interaction"):
        get_lric_divergence(
            adata,
            "cross_pcf",
            feature_a={"source": src},
            feature_b={"source": src},
        )
    with pytest.raises(ValueError, match="No rows match"):
        get_lric_divergence(
            adata,
            "cross_pcf",
            feature_a={"interaction": "nope^nope"},
            feature_b={"interaction": "nope^nope"},
        )
    with pytest.raises(ValueError, match="shared finite bins"):
        get_lric_divergence(
            adata,
            "cross_pcf",
            feature_a={"interaction": adata.uns["cross_pcf"]["interaction"].unique()[0]},
            feature_b={"interaction": adata.uns["cross_pcf"]["interaction"].unique()[1]},
            min_bins=99,
        )
