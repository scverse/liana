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


def test_translate_column_deduplicates_map_df() -> None:
    """A duplicated `map_df` row used to count as a second ortholog and drop the gene."""
    map_df = pd.DataFrame({"source": ["CD8A", "CD8A", "CD8B"], "target": ["Cd8a", "Cd8a", "Cd8b1"]})
    df = pd.DataFrame({"symbol": ["CD8A_CD8B"]})

    translated = translate_column(df, map_df=map_df, column="symbol")

    assert translated["symbol"].tolist() == ["Cd8a_Cd8b1"]


def test_translate_resource_warns_when_empty() -> None:
    """Nothing translated (here: a case mismatch) used to return an empty frame silently."""
    resource = select_resource("consensus").head(3)
    map_df = pd.DataFrame({"source": ["lgals9", "ptprc"], "target": ["Lgals9", "Ptprc"]})

    with pytest.warns(UserWarning, match="No interactions were translated"):
        translated = translate_resource(resource, map_df)

    assert translated.empty


def _hcop_gz() -> bytes:
    import gzip

    table = "human_symbol\tmouse_symbol\tsupport\nCD8A\tCd8a\tHGNC,Inparanoid\n"
    return gzip.compress(table.encode())


@pytest.mark.parametrize("retry_readable", [True, False])
def test_get_hcop_retries_unreadable_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, retry_readable: bool
) -> None:
    """A truncated cached file is not hash-checked; it used to be served unvalidated forever."""
    import pooch

    content = _hcop_gz()
    cached = tmp_path / "human_mouse_hcop_fifteen_column.txt.gz"
    truncated = content[: len(content) // 2]
    cached.write_bytes(truncated)
    payload = content if retry_readable else content[:10]

    def _fake_retrieve(url: str, known_hash: str | None, fname: str, path: pathlib.Path) -> str:
        target = pathlib.Path(path) / fname
        target.write_bytes(payload)
        return str(target)

    monkeypatch.setattr(pooch, "retrieve", _fake_retrieve)

    if retry_readable:
        with pytest.warns(UserWarning, match="could not be read"):
            mapping = get_hcop_orthologs(filename=cached, min_evidence=0)
        assert mapping["human_symbol"].tolist() == ["CD8A"]
        assert cached.read_bytes() == content
    else:
        with pytest.warns(UserWarning, match="could not be read"), pytest.raises(OSError, match="downloading it again"):
            get_hcop_orthologs(filename=cached, min_evidence=0)
        # a failed retry must not cost the user their cache
        assert cached.read_bytes() == truncated
