"""Fixtures for the tests that need liana's external resources.

These are the only tests that reach the network, and they cache what they download under ``tests/.cache``.

CI runs the suite with ``pytest -n auto``, which gives every xdist worker its own session and so its own copy of these session-scoped fixtures.
Both downloads go through :func:`pooch.retrieve`, which writes to a temporary file and moves it into place, so a worker sees either the finished file or none at all.
"""

from __future__ import annotations

import pathlib

import pytest


@pytest.fixture(scope="session")
def download_cache() -> pathlib.Path:
    """Directory the tests download their (large) external files to, once."""
    cache = pathlib.Path(__file__).parents[1] / ".cache"
    cache.mkdir(exist_ok=True)

    return cache


@pytest.fixture(scope="session")
def metalinks_db(download_cache: pathlib.Path) -> str:
    """Path to MetaLinksDB, downloaded on first use."""
    from liana.resource.get_metalinks import _download_metalinksdb

    return str(_download_metalinksdb(cache_dir=download_cache, verbose=False))


@pytest.fixture(scope="session")
def hcop_file(download_cache: pathlib.Path) -> str:
    """Path the human-mouse HCOP table is cached at, downloaded on first use."""
    from liana.resource import get_hcop_orthologs

    path = download_cache / "human_mouse_hcop_fifteen_column.txt.gz"
    get_hcop_orthologs(target_organism="mouse", filename=path, min_evidence=0)

    return str(path)
