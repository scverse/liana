import pathlib

import pandas as pd
import pytest

from liana.resource import get_hcop_orthologs, select_resource, translate_column, translate_resource


def test_complex_cases() -> None:
    map_df = pd.DataFrame(
        {
            "source": ["CSF2RA", "IFNL3", "IFNL3", "IFNLR1", "IL10RB", "HCST", "CD8A", "CD8B", "IL4"],
            "target": ["Csf2ra", "Ifnl3", "Ifnl2", "Ifnlr1", "Il10rb", "Hcst", "Cd8a", "Cd8b1", "Il4"],
        }
    )
    df = pd.DataFrame(
        {
            "symbol": [
                "CSF2RA_CSF2RB",  # one to many
                "IFNL3_IFNLR1_IL10RB",  # 3 subunits
                "HCST_KLRK1",  # one subunit missing
                "CD8A_CD8B",  # 1 to 1
                "IL4",  # 1 to 1 simple protein
            ]
        }
    )

    default = translate_column(
        df,
        map_df=map_df,
        column="symbol",
    )
    assert all(default["symbol"] == ["Cd8a_Cd8b1", "Il4"])

    to_many = translate_column(
        df,
        map_df=map_df,
        column="symbol",
        replace=True,
        one_to_many=2,
    )
    expected = [
        "Cd8a_Cd8b1",
        "Ifnl2_Ifnlr1_Il10rb",
        "Ifnl3_Ifnlr1_Il10rb",
        "Il4",
    ]

    assert to_many.shape == (4, 1)
    assert all(to_many["symbol"].isin(expected))

    keep_missing = translate_column(
        df,
        map_df=map_df,
        column="symbol",
        replace=False,
        one_to_many=2,
    )
    untranslated = keep_missing["symbol"].isin(["HCST_KLRK1"])
    assert untranslated.any()


@pytest.mark.network
def test_translate_resource(hcop_file: str) -> None:
    resource = select_resource()
    map_df = get_hcop_orthologs(
        target_organism="mouse", filename=hcop_file, columns=["human_symbol", "mouse_symbol"], min_evidence=3
    )
    map_df = map_df.rename(columns={"human_symbol": "source", "mouse_symbol": "target"})

    translated = translate_resource(resource, map_df, one_to_many=1)
    assert translated.shape[0] > 3000
    translated2 = translate_resource(resource, map_df, one_to_many=5, replace=False)
    assert translated2.shape[0] > translated.shape[0]


@pytest.mark.network
def test_get_hcop(hcop_file: str) -> None:
    mapping = get_hcop_orthologs(filename=hcop_file, columns=None, min_evidence=0)
    assert mapping.shape[0] > 1000
    assert mapping.shape[1] == 16  # 15 columns + added evidence column


@pytest.mark.network
def test_get_hcop_caches_under_datasetdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, hcop_file: str
) -> None:
    """Omitting `filename` downloads to a file named after the URL, under scanpy's dataset directory."""
    import shutil

    import pooch
    import scanpy as sc

    def _fake_retrieve(url: str, known_hash: str | None, fname: str, path: pathlib.Path) -> str:
        target = pathlib.Path(path) / fname
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hcop_file, target)
        return str(target)

    monkeypatch.setattr(pooch, "retrieve", _fake_retrieve)

    original = sc.settings.datasetdir
    sc.settings.datasetdir = tmp_path
    try:
        derived = get_hcop_orthologs(columns=None, min_evidence=0)
    finally:
        sc.settings.datasetdir = original

    assert [p.name for p in tmp_path.iterdir()] == ["human_mouse_hcop_fifteen_column.txt.gz"]
    pd.testing.assert_frame_equal(derived, get_hcop_orthologs(filename=hcop_file, columns=None, min_evidence=0))
