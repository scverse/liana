import pytest

from liana.resource.select_resource import _handle_resource, select_resource


def test_select_interactions() -> None:
    # generate list of 2x strings
    interactions = [("a", "b"), ("c", "d")]

    resource = _handle_resource(
        interactions=interactions,
        x_name="x",
        y_name="y",
        verbose=True,
        # These should be ignored
        resource=select_resource("consensus"),
        resource_name="consensus",
    )

    assert resource.shape[0] == 2
    assert (resource.columns == ["x", "y"]).all()


def test_select_resource() -> None:
    resource = _handle_resource(
        interactions=None,
        x_name="ligand",
        y_name="receptor",
        verbose=True,
        resource=select_resource("consensus"),
        # This should be ignored
        resource_name="ignore me",
    )

    assert resource.shape[0] == 4620
    assert (resource.columns == ["ligand", "receptor"]).all()


def test_select_resource_name() -> None:
    resource = _handle_resource(
        interactions=None,
        resource=None,
        resource_name="cellchatdb",
        x_name="x",
        y_name="y",
        verbose=True,
    )

    assert resource.shape[0] == 1912
    assert (resource.columns == ["ligand", "receptor"]).all()


def test_consensus_pecam1_cd38_direction() -> None:
    # https://github.com/scverse/liana/issues/218
    # The PECAM1-CD38 interaction is directed PECAM1 (ligand) -> CD38 (receptor),
    # as in CellPhoneDB and the literature (PMID: 7542249). The consensus row was
    # flipped (CD38 -> PECAM1); assert the corrected direction is the only one.
    resource = select_resource("consensus")
    pair = resource[resource[["ligand", "receptor"]].isin(["CD38", "PECAM1"]).all(axis=1)]
    assert pair[["ligand", "receptor"]].to_numpy().tolist() == [["PECAM1", "CD38"]]


def test_consensus_excludes_smad3_receptor() -> None:
    # https://github.com/scverse/liana/issues/218
    # SMAD3 is an intracellular transcription factor, not a membrane receptor,
    # so it must not appear as a receptor in the consensus resource.
    resource = select_resource("consensus")
    assert not (resource["receptor"] == "SMAD3").any()


def test_resource_exception_none() -> None:
    with pytest.raises(ValueError):
        _handle_resource(
            interactions=None,
            resource=None,
            resource_name=None,
            x_name="x",
            y_name="y",
            verbose=True,
        )


def test_resource_exception_xy() -> None:
    with pytest.raises(ValueError):
        _handle_resource(
            interactions=None,
            resource=select_resource("consensus"),
            resource_name=None,
            x_name="x",
            y_name="y",
            verbose=True,
        )
