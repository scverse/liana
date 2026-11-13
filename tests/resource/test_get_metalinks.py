import pathlib
import sqlite3

import pandas as pd
import pytest

from liana.resource.get_metalinks import describe_metalinks, get_metalinks, get_metalinks_values


@pytest.mark.network
def test_get_metalinks(metalinks_db: str) -> None:
    result = get_metalinks(db_path=metalinks_db, tissue_location="Brain", hmdb_ids="HMDB0000073", uniprot_ids="P14416")
    assert isinstance(result, pd.DataFrame)
    # one row per curating `source`; this edge is annotated by five databases, with `mor` 0 and 1
    assert result.shape == (10, 8)
    assert result["source"].nunique() == 5
    assert set(result["mor"]) == {0, 1}
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


@pytest.fixture
def toy_metalinks_db(tmp_path: pathlib.Path) -> str:
    """A minimal MetalinksDB with the tables :func:`get_metalinks` queries, so `split` is tested offline."""
    path = tmp_path / "metalinks.db"
    edges = pd.DataFrame(
        [
            # the same lr edge from two curators -> one resource row
            ("H1", "P1", 1, None, "lr", "CellPhoneDB"),
            ("H1", "P1", 1, None, "lr", "scConnect"),
            ("H2", "P2", 1, None, "lr", "Cellinker"),
            # production and degradation from two curators -> weight 0, kept
            ("H1", "P3", 1, None, "pd", "recon"),
            ("H1", "P3", -1, None, "pd", "rhea"),
            # export -> +1, import -> -1
            ("H1", "P4", 1, "out", "pd", "recon"),
            ("H2", "P4", -1, "in", "pd", "recon"),
            # out and in -> mean 0, dropped from t_net
            ("H2", "P5", 1, "out", "pd", "recon"),
            ("H2", "P5", 1, "in", "pd", "hmr"),
            # unknown direction -> dropped from t_net
            ("H2", "P3", 1, "both", "pd", "hmdb"),
        ],
        columns=["hmdb", "uniprot", "mor", "transport_direction", "type", "source"],
    )
    metabolites = pd.DataFrame({"hmdb": ["H1", "H2"], "metabolite": ["M1", "M2"]})
    proteins = pd.DataFrame({"uniprot": [f"P{i}" for i in range(1, 6)], "gene_symbol": [f"G{i}" for i in range(1, 6)]})
    with sqlite3.connect(path) as conn:
        edges.to_sql("edges", conn, index=False)
        metabolites.to_sql("metabolites", conn, index=False)
        proteins.to_sql("proteins", conn, index=False)
    return str(path)


def test_get_metalinks_split(toy_metalinks_db: str) -> None:
    resource, pd_net, t_net = get_metalinks(db_path=toy_metalinks_db, split=True)

    pd.testing.assert_frame_equal(resource, pd.DataFrame({"ligand": ["M1", "M2"], "receptor": ["G1", "G2"]}))
    pd.testing.assert_frame_equal(
        pd_net,
        pd.DataFrame(
            {
                "source": ["M1", "M1", "M2", "M2", "M2"],
                "target": ["G3", "G4", "G3", "G4", "G5"],
                "weight": [0.0, 1.0, 1.0, -1.0, 1.0],
            }
        ),
    )
    assert t_net is not None
    pd.testing.assert_frame_equal(
        t_net, pd.DataFrame({"source": ["M1", "M2"], "target": ["G4", "G4"], "weight": [1.0, -1.0]})
    )


def test_get_metalinks_split_without_transport(toy_metalinks_db: str) -> None:
    _, _, t_net = get_metalinks(db_path=toy_metalinks_db, uniprot_ids=["P1", "P3"], split=True)
    assert t_net is None


def test_get_metalinks_split_requires_lr_and_pd(toy_metalinks_db: str) -> None:
    with pytest.raises(ValueError, match="'lr' and 'pd'"):
        get_metalinks(db_path=toy_metalinks_db, types=["lr"], split=True)
