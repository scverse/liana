import pathlib

import pandas as pd
import pytest

from liana.resource.get_metalinks import describe_metalinks, get_metalinks, get_metalinks_values


@pytest.mark.network
def test_get_metalinks(metalinks_db: str) -> None:
    result = get_metalinks(
        db_path=metalinks_db, tissue_location="Brain", hmdb_ids="HMDB0000073", uniprot_ids="P14416"
    ).drop_duplicates(["hmdb", "uniprot"])
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 8)
    assert "Dopamine" in result["metabolite"].to_numpy()
    assert "HMDB0000073" in result["hmdb"].to_numpy()
    assert "P14416" in result["uniprot"].to_numpy()
    assert "DRD2" in result["gene_symbol"].to_numpy()


@pytest.mark.network
def test_get_metalinks_values(metalinks_db: str) -> None:
    result = get_metalinks_values("disease", "disease", db_path=metalinks_db)

    # Check that the result is a list
    assert isinstance(result, list)

    assert "Obesity" in result
    assert "Schizophrenia" in result
    assert len(result) == 567


@pytest.mark.network
def test_describe_metalinks(metalinks_db: str) -> None:
    out = describe_metalinks(db_path=metalinks_db, return_output=True)
    assert out is not None
    assert "metabolites" in out
    assert "proteins" in out
    assert "edges" in out
    assert "Column ID: 8" in out


@pytest.mark.network
def test_metalinks_resolves_db_when_no_path_given(
    monkeypatch: pytest.MonkeyPatch, download_cache: pathlib.Path, metalinks_db: str
) -> None:
    """Omitting `db_path` fetches the database into `scanpy.settings.datasetdir` and reuses it from there."""
    monkeypatch.chdir(download_cache)

    pd.testing.assert_frame_equal(
        get_metalinks(tissue_location="Brain"), get_metalinks(db_path=metalinks_db, tissue_location="Brain")
    )
    assert get_metalinks_values("disease", "disease") == get_metalinks_values(
        "disease", "disease", db_path=metalinks_db
    )
    assert describe_metalinks(return_output=True) == describe_metalinks(db_path=metalinks_db, return_output=True)
