import pathlib
from itertools import product

import numpy as np
import pytest
from anndata import AnnData
from fast_array_utils.conv import to_dense
from pandas import DataFrame, read_csv
from pandas.testing import assert_frame_equal
from tests._helpers import as_frame, get_csr, get_obs, get_raw_csr

from liana._core._constants import DefaultValues as V
from liana.method.sc._liana_pipe import _calc_log2fc, _expm1_base, liana_pipe

groupby = "bulk_labels"


@pytest.fixture
def pbmc68k() -> AnnData:
    # pbmc68k_reduced ships scaled data in .X and log-norm in .raw; put log-norm in
    # .X so the default (use_raw=False) path is exercised against the reference output.
    from scanpy.datasets import pbmc68k_reduced

    adata = pbmc68k_reduced()
    adata.X = get_raw_csr(adata).copy()
    return adata


# Test ALL Default parameters
def test_liana_pipe_defaults(pbmc68k: AnnData, data_dir: pathlib.Path) -> None:
    all_defaults = as_frame(
        liana_pipe(
            adata=pbmc68k,
            groupby=groupby,
            resource_name=V.resource_name,
            groupby_pairs=V.groupby_pairs,
            expr_prop=V.expr_prop,
            min_cells=V.min_cells,
            de_method=V.de_method,
            base=V.logbase,
            n_perms=V.n_perms,
            seed=V.seed,
            verbose=V.verbose,
            supp_columns=[],
            resource=V.resource,
            use_raw=False,  # local fixture puts log-norm in .X
            layer=V.layer,
            n_jobs=1,
            interactions=V.interactions,
        )
    )

    assert "prop_min" in all_defaults.columns
    all_defaults = all_defaults.sort_values(by=list(all_defaults.columns))

    exp_defaults = read_csv(data_dir / "all_defaults.csv", index_col=0)
    exp_defaults = exp_defaults.sort_values(by=list(all_defaults.columns))
    exp_defaults.index = all_defaults.index

    assert_frame_equal(
        all_defaults, exp_defaults, check_dtype=False, check_exact=False, check_index_type=False, rtol=1e-3
    )


# A heteromeric complex scores by its *limiting* subunit -- `return_all_lrs` used to
# collapse the exploded subunits by resource order before that reduction could run.
SOURCE, TARGET = "CD14+ Monocyte", "CD8+ Cytotoxic T"
COMPLEX = "CD74_CXCR4"


def _group_stats(adata: AnnData, gene: str, label: str) -> tuple[float, float]:
    """Mean expression of `gene` over the cells of one group, and the fraction expressing it."""
    column = to_dense(get_csr(adata[:, gene])).ravel()
    values = column[get_obs(adata)[groupby].to_numpy() == label]
    return float(values.mean()), float((values > 0).mean())


@pytest.mark.parametrize("return_all_lrs", [False, True])
def test_complex_reduces_to_min_subunit(pbmc68k: AnnData, return_all_lrs: bool) -> None:
    expr_prop = 0.2

    # ground truth, straight off the matrix: what each subunit is worth on its own
    subunits = {name: _group_stats(pbmc68k, name, TARGET) for name in COMPLEX.split("_")}
    ligand_mean, ligand_prop = _group_stats(pbmc68k, "MIF", SOURCE)

    limiting = min(subunits, key=lambda name: subunits[name][0])
    first = COMPLEX.split("_")[0]
    # the two subunits have to disagree, and the first-listed one has to be the wrong
    # answer, or the assertions below cannot tell the two code paths apart
    assert len({mean for mean, _ in subunits.values()}) == len(subunits)
    assert first != limiting

    lr_res = as_frame(
        liana_pipe(
            adata=pbmc68k,
            groupby=groupby,
            resource_name=V.resource_name,
            expr_prop=expr_prop,
            groupby_pairs=V.groupby_pairs,
            min_cells=V.min_cells,
            de_method="wilcoxon",
            base=V.logbase,
            n_perms=None,
            seed=V.seed,
            verbose=V.verbose,
            supp_columns=[],
            resource=V.resource,
            use_raw=False,
            layer=V.layer,
            return_all_lrs=return_all_lrs,
            n_jobs=1,
            interactions=V.interactions,
        )
    )

    match = lr_res[
        (lr_res["ligand_complex"] == "MIF")
        & (lr_res["receptor_complex"] == COMPLEX)
        & (lr_res["source"] == SOURCE)
        & (lr_res["target"] == TARGET)
    ]
    assert len(match) == 1
    row = match.iloc[0]

    # the complex is represented by its weakest subunit, not by whichever is listed first
    assert row["receptor"] == limiting
    assert row["receptor_means"] == pytest.approx(subunits[limiting][0], rel=1e-3)
    assert row["receptor_props"] == pytest.approx(subunits[limiting][1], rel=1e-3)
    assert row["receptor_means"] != pytest.approx(subunits[first][0], rel=1e-3)

    # the ligand is a single subunit, so it is carried through untouched
    assert row["ligand"] == "MIF"
    assert row["ligand_means"] == pytest.approx(ligand_mean, rel=1e-3)

    # the expr_prop gate reads every subunit's *proportion*, both sides of the interaction
    props = [ligand_prop] + [prop for _, prop in subunits.values()]
    assert row["prop_min"] == pytest.approx(min(props), rel=1e-3)
    if return_all_lrs:
        assert bool(row["lrs_to_keep"]) is (min(props) >= expr_prop)

    # and the values themselves, so a shift in the fixture or in how the means are
    # computed surfaces here rather than silently redefining what the test compares
    assert {name: mean for name, (mean, _) in subunits.items()} == pytest.approx(
        {"CD74": 2.331778, "CXCR4": 0.705593}, rel=1e-3
    )
    assert {name: prop for name, (_, prop) in subunits.items()} == pytest.approx(
        {"CD74": 0.870370, "CXCR4": 0.370370}, rel=1e-3
    )
    assert (ligand_mean, ligand_prop) == pytest.approx((0.521814, 0.325581), rel=1e-3)


# Test NOT Default parameters
def test_liana_pipe_not_defaults(pbmc68k: AnnData, data_dir: pathlib.Path) -> None:
    not_defaults = as_frame(
        liana_pipe(
            adata=pbmc68k,
            groupby=groupby,
            resource_name=V.resource_name,
            expr_prop=0.2,
            groupby_pairs=V.groupby_pairs,
            min_cells=V.min_cells,
            de_method="wilcoxon",
            base=V.logbase,
            n_perms=V.n_perms,
            seed=V.seed,
            verbose=V.verbose,
            supp_columns=["ligand_pvals", "receptor_pvals"],
            resource=V.resource,
            use_raw=False,  # local fixture puts log-norm in .X
            layer=V.layer,
            return_all_lrs=True,
            n_jobs=1,
            interactions=V.interactions,
        )
    )

    assert all(np.isin(["lrs_to_keep"], not_defaults.columns))
    assert all(np.isin(["ligand_pvals", "receptor_pvals"], not_defaults.columns))
    not_defaults = not_defaults.sort_values(list(not_defaults.columns))

    exp_defaults = read_csv(data_dir / "not_defaults.csv", index_col=0)
    exp_defaults = exp_defaults.sort_values(list(not_defaults.columns))
    exp_defaults.index = not_defaults.index
    assert_frame_equal(
        not_defaults, exp_defaults, check_dtype=False, check_index_type=False, check_exact=False, rtol=1e-3
    )


def test_liana_pipe_subset(pbmc68k: AnnData) -> None:
    cts = ["CD34+", "Dendritic", "CD56 NK", "CD19+ B"]
    pairs = DataFrame(list(product(cts, cts)), columns=["source", "target"])
    groupby_pairs = pairs[pairs["source"] == "Dendritic"]

    subset = as_frame(
        liana_pipe(
            adata=pbmc68k,
            groupby=groupby,
            resource_name=V.resource_name,
            expr_prop=0.05,
            groupby_pairs=groupby_pairs,
            min_cells=V.min_cells,
            de_method=V.de_method,
            base=V.logbase,
            n_perms=V.n_perms,
            seed=V.seed,
            verbose=V.verbose,
            resource=V.resource,
            use_raw=False,  # local fixture puts log-norm in .X
            layer=V.layer,
            n_jobs=1,
            interactions=V.interactions,
        )
    )

    assert subset.shape == (46, 23)


def test_expm1_fun(pbmc68k: AnnData) -> None:
    data = get_raw_csr(pbmc68k).data
    expm1_mat = _expm1_base(data, V.logbase)
    # `logbase` is `e`, so this is `np.expm1`
    np.testing.assert_allclose(expm1_mat, np.expm1(data), rtol=1e-6)
    np.testing.assert_almost_equal(np.sum(expm1_mat), 1386299.6, decimal=1)


def test_calc_log2fc(pbmc68k: AnnData) -> None:
    # the arguments used to be swapped, so this exercised `data ** e`
    normcounts = get_raw_csr(pbmc68k).copy()
    normcounts.data = _expm1_base(normcounts.data, V.logbase)
    pbmc68k.layers["normcounts"] = normcounts
    get_obs(pbmc68k)["@label"] = get_obs(pbmc68k)["bulk_labels"]
    np.testing.assert_almost_equal(np.mean(_calc_log2fc(pbmc68k, "Dendritic")), -0.094648157)


def test_calc_log2fc_no_rest_raises(pbmc68k: AnnData) -> None:
    single_label = pbmc68k[get_obs(pbmc68k)["bulk_labels"] == "Dendritic"].copy()
    normcounts = get_raw_csr(single_label).copy()
    normcounts.data = _expm1_base(normcounts.data, V.logbase)
    single_label.layers["normcounts"] = normcounts
    get_obs(single_label)["@label"] = get_obs(single_label)["bulk_labels"]

    with pytest.raises(ValueError, match="Cannot compute log2FC"):
        _calc_log2fc(single_label, "Dendritic")
