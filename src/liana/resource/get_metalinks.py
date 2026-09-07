import sqlite3
from pathlib import Path

import pandas as pd
import pooch
import scanpy as sc

from liana._core._common import _logg

_METALINKS_URL = "https://github.com/scverse/liana/releases/download/metalinksdb/metalinksdb.db"
_METALINKS_HASH = "sha256:84df58e659cd0fe318b10f6d5d7ac1f16806b1088701fccf4260e78ba0982513"


def _download_metalinksdb(cache_dir: str | Path | None = None, verbose: bool = True) -> Path:
    """
    Ensures the Metalinksdb is downloaded and available for use.

    Parameters
    ----------
    cache_dir
        Directory the database is cached in, defaulting to :attr:`scanpy.settings.datasetdir`.
    verbose
        Verbosity flag.

    Returns
    -------
    The path to the downloaded database file.
    """
    _logg("Retrieving database...", verbose=verbose)

    return Path(
        pooch.retrieve(
            _METALINKS_URL,
            known_hash=_METALINKS_HASH,
            fname="metalinksdb.db",
            path=sc.settings.datasetdir if cache_dir is None else cache_dir,
            progressbar=verbose,
        )
    )


def _format_clauses(
    input_data: list[str] | None,
    column_name: str,
    table_ref: str,
    where_clauses: list[str],
) -> None:
    if input_data:
        formatted_str = ", ".join([f"'{i}'" for i in input_data])
        where_clauses.append(f"{table_ref}.{column_name} IN ({formatted_str})")


def get_metalinks(
    db_path: str | Path | None = None,
    types: str | list[str] | None = None,
    cell_location: str | list[str] | None = None,
    tissue_location: str | list[str] | None = None,
    biospecimen_location: str | list[str] | None = None,
    disease: str | list[str] | None = None,
    pathway: str | list[str] | None = None,
    hmdb_ids: str | list[str] | None = None,
    uniprot_ids: str | list[str] | None = None,
    source: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetches edges of metabolite-proteins with specified annotations, applying filters if they are not None.

    Allows filtering by lists of hmdb and uniprot IDs and avoids duplicate column names, and returns the results as a pandas DataFrame.
    Filters are applied using INNER JOINs and WHERE clauses - i.e. the results are the intersection of the filters.

    Parameters
    ----------
    db_path
        Path to the SQLite database file. If None, the database will be downloaded to the current working directory.
    types
        Desired edge types. Options are: ['lr', 'pd'], where 'lr' stands for 'ligand-receptor' and 'pd' stands for 'production-degradation'.
    cell_location
        Desired metabolite cell locations.
    tissue_location
        Desired metabolite tissue locations.
    biospecimen_location
        Desired metabolite biospecimen locations.
    disease
        Desired metabolite diseases.
    pathway
        Desired metabolite pathways.
    hmdb_ids
        Desired HMDB IDs.
    uniprot_ids
        Desired UniProt IDs.
    source
        Desired source databases.

    Returns
    -------
    A pandas DataFrame containing the query results without the source column.

    Examples
    --------
    This function downloads MetalinksDB on first use, so it is not run here.
    Metabolite-receptor edges restricted to secreted metabolites are obtained
    with::

        resource = get_metalinks(
            types=["lr"],
            biospecimen_location="Blood",
        )
    """
    path = Path(db_path) if db_path is not None else _download_metalinksdb()
    conn = sqlite3.connect(path)

    # Adjusted SELECT statement to exclude the source column
    base_query = """
    SELECT DISTINCT e.hmdb as hmdb,
                e.uniprot AS uniprot,
                p.gene_symbol as gene_symbol,
                m.metabolite AS metabolite,
                e.mor as mor,
                e.transport_direction as transport_direction,
                e.type AS type,
                e.source AS source
    FROM edges e
    LEFT JOIN metabolites m ON e.hmdb = m.hmdb
    LEFT JOIN proteins p ON e.uniprot = p.uniprot
    """

    def _to_list(x: str | list[str] | None) -> list[str] | None:
        if isinstance(x, str):
            return [x]
        return x

    cell_location = _to_list(cell_location)
    tissue_location = _to_list(tissue_location)
    biospecimen_location = _to_list(biospecimen_location)
    disease = _to_list(disease)
    pathway = _to_list(pathway)
    hmdb_ids = _to_list(hmdb_ids)
    uniprot_ids = _to_list(uniprot_ids)
    types = _to_list(types)
    source = _to_list(source)

    annotations_filters = {
        "cell_location": cell_location,
        "tissue_location": tissue_location,
        "biospecimen_location": biospecimen_location,
        "disease": disease,
        "pathway": pathway,
    }

    join_clauses = []
    where_clauses = []
    for annotation_table, values in annotations_filters.items():
        if values is not None:
            join_clause = f"INNER JOIN {annotation_table} ON m.hmdb = {annotation_table}.hmdb"
            join_clauses.append(join_clause)

            values_str = ", ".join([f"'{value}'" for value in values])
            where_clause = f"{annotation_table}.{annotation_table} IN ({values_str})"
            where_clauses.append(where_clause)

    _format_clauses(types, "type", "e", where_clauses)
    _format_clauses(hmdb_ids, "hmdb", "m", where_clauses)
    _format_clauses(uniprot_ids, "uniprot", "p", where_clauses)
    _format_clauses(source, "source", "e", where_clauses)

    full_query = base_query
    if join_clauses:
        full_query += " " + " ".join(join_clauses)
    if where_clauses:
        full_query += " WHERE " + " AND ".join(where_clauses)

    df = pd.read_sql_query(full_query, conn)
    conn.close()

    return df


def get_metalinks_values(table_name: str, column_name: str, db_path: str | None = None) -> list[str]:
    """
    Fetches distinct values from a specified column in a specified table.

    Parameters
    ----------
    table_name
        Name of the table from which to fetch distinct values.
    column_name
        Name of the column from which to fetch distinct values.
    db_path
        Path to the SQLite database file. If None, the database will be downloaded to the current working directory.

    Returns
    -------
    A list of distinct values from the specified column.

    Examples
    --------
    This function downloads MetalinksDB on first use, so it is not run here. Use
    it to discover the values accepted by the filters of
    :func:`liana.rs.get_metalinks`::

        get_metalinks_values(table_name="tissue_location", column_name="tissue_location")
    """
    path = Path(db_path) if db_path is not None else _download_metalinksdb()
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    query = f"SELECT DISTINCT {column_name} FROM {table_name};"
    cursor.execute(query)
    distinct_values = cursor.fetchall()
    conn.close()
    return [value[0] for value in distinct_values]


def describe_metalinks(db_path: str | None = None, return_output: bool = False) -> str | None:
    """
    Prints the schema information and foreign key details for all tables in the specified SQLite database.

    Parameters
    ----------
    db_path
        Path to the SQLite database file. If None, the database will be downloaded to the current working directory.
    return_output
        Whether to return the output or just print it.

    Returns
    -------
    The database schema description.

    Examples
    --------
    This function downloads MetalinksDB on first use, so it is not run here.
    It prints the tables and columns that :func:`liana.rs.get_metalinks` and :func:`liana.rs.get_metalinks_values` query::

        describe_metalinks()
    """
    path = Path(db_path) if db_path is not None else _download_metalinksdb()
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    output = ""
    for table in tables:
        table_name = table[0]
        output += f"Schema of table: {table_name}\n{'=' * len(f'Schema of table: {table_name}')}\n"

        cursor.execute(f"PRAGMA table_info({table_name});")
        schema_info = cursor.fetchall()
        for column in schema_info:
            cid, name, ctype, _, _, pk = column
            output += f"Column ID: {cid}, Name: {name}, Type: {ctype}, Primary Key: {pk}\n"

        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        fk_info = cursor.fetchall()
        if fk_info:
            output += "\nForeign Keys:\n"
            for fk in fk_info:
                id, seq, table, from_col, to_col, _, _, _ = fk
                output += f"ID: {id}, Seq: {seq}, Table: {table}, From: {from_col}, To: {to_col}\n"
        else:
            output += "\nNo Foreign Keys.\n"
        output += "-" * 40 + "\n"

    if return_output:
        return output
    else:
        print(output)
        return None
