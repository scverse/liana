from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import scanpy as sc

_HCOP_BASE = "https://storage.googleapis.com/public-download-files/hcop"


def _replace_subunits(
    lst: list[str],
    my_dict: dict[str, str | list[str]],
    one_to_many: int,
) -> list[list[str] | float]:
    # NaN marks a subunit with no (or too many) orthologs; the caller drops those rows.
    result: list[list[str] | float] = []
    for x in lst:
        if x in my_dict:
            value = my_dict[x]

            if not isinstance(value, list):
                value = [value]

            if len(value) > one_to_many:
                result.append(np.nan)
            else:
                result.append(value)
        else:
            result.append(np.nan)
    return result


def _generate_orthologs(
    data: pd.DataFrame,
    column: str,
    map_dict: dict[str, str | list[str]],
    one_to_many: int,
) -> pd.DataFrame:
    df = data[[column]].drop_duplicates().set_index(column)

    df["subunits"] = df.index.str.split("_")
    df["subunits"] = df["subunits"].apply(
        _replace_subunits,
        args=(
            map_dict,
            one_to_many,
        ),
    )
    df = df["subunits"].explode().reset_index()

    grouped = df.groupby(column).filter(lambda x: x["subunits"].notna().all()).groupby(column)

    # Generate all possible subunit combinations within each group
    complexes = []
    for name, group in grouped:
        if group["subunits"].isnull().all():
            continue
        subunit_lists = [list(x) for x in group["subunits"]]
        complex_combinations = list(product(*subunit_lists))
        for complex in complex_combinations:
            complexes.append((name, "_".join(complex)))

    # Create output DataFrame
    col_names = ["orthology_source", "orthology_target"]
    result = pd.DataFrame(complexes, columns=col_names).set_index("orthology_source")

    return result


def translate_column(
    resource: pd.DataFrame,
    map_df: pd.DataFrame,
    column: str,
    replace: bool = True,
    one_to_many: int = 1,
) -> pd.DataFrame:
    """
    Generate orthologs for a given column in a DataFrame.

    Parameters
    ----------
    resource
        Input DataFrame.
    map_df
        DataFrame with orthology mappings, where the first column is the source and the second column is the target for mapping.
    column
        Column name to translate.
    replace
        Whether to replace the original column with the translated values. Default is True.
        If False, it will create a new column with the prefix ``orthology_``.
    one_to_many
        Maximum number of orthologs allowed per gene. Default is 1.

    Notes
    -----
    This function generates orthologs for a given column in a DataFrame.
    It handles complex names by splitting them into subunits and generating all possible combinations of orthologs.
    It assumes that subunits are separated by an underscore ("_").

    Returns
    -------
    Resulting DataFrame with translated column.

    Raises
    ------
    ValueError
        If the `mapping_df` does not contain 'source' and 'target' columns or `one_to_many` is not an integer

    Examples
    --------
    `map_df` maps human symbols (`source`) to the target organism (`target`).
    :func:`liana.rs.get_hcop_orthologs` builds one; it is written out here to keep the example offline:

    >>> import pandas as pd
    >>> import liana as li
    >>> resource = li.rs.select_resource("consensus").head(3)
    >>> map_df = pd.DataFrame(
    ...     {"source": ["LGALS9", "PTPRC", "MET", "CD44"], "target": ["Lgals9", "Ptprc", "Met", "Cd44"]}
    ... )
    >>> li.rs.translate_column(resource, map_df, column="ligand")
       ligand receptor
    0  Lgals9    PTPRC
    1  Lgals9      MET
    2  Lgals9     CD44

    With `replace=False` the translation is added as an `orthology_ligand` column
    instead of overwriting `ligand`. Use
    :func:`liana.rs.translate_resource` to do both sides at once.
    """
    if not isinstance(one_to_many, int):
        raise ValueError("`one_to_many` should be a positive integer!")
    if ["source", "target"] != map_df.columns.tolist():
        raise ValueError("The `map_df` DataFrame must have two columns named 'source' and 'target'!")

    # get orthologs
    map_df = map_df.set_index("source")
    map_dict: dict[str, str | list[str]] = {
        str(source): list(targets) for source, targets in map_df.groupby(level=0)["target"]
    }
    map_data = _generate_orthologs(resource, column, map_dict, one_to_many)

    # join orthologs
    resource = resource.merge(map_data, left_on=column, right_index=True, how="left")

    # replace orthologs
    if replace:
        resource[column] = resource["orthology_target"]
    else:
        resource[f"orthology_{column}"] = resource.apply(
            lambda x: x["orthology_target"] if not pd.isnull(x["orthology_target"]) else x[column],
            axis=1,
        )
    resource = resource.drop(columns=["orthology_target"])

    resource = resource.dropna(subset=[column])
    return resource


# function that loops over columns and applies translate_column
def translate_resource(
    resource: pd.DataFrame,
    map_df: pd.DataFrame,
    columns: list[str] | None = None,
    replace: bool = True,
    one_to_many: int = 1,
) -> pd.DataFrame:
    """
    Generate orthologs for multiple columns in a DataFrame.

    Parameters
    ----------
    resource
        Input DataFrame.
    map_df
        DataFrame with orthology mappings, where the first column is the source and the second column is the target for mapping.
    columns
        List of column names to translate.
    **kwargs
        Additional arguments for `liana.rs.translate_column`.

    Returns
    -------
    Resulting DataFrame with translated columns.

    Examples
    --------
    Translates the `'ligand'` and `'receptor'` columns in one go.
    In practice `map_df` comes from :func:`liana.rs.get_hcop_orthologs`; interactions whose partners have no ortholog are dropped:

    >>> import pandas as pd
    >>> import liana as li
    >>> resource = li.rs.select_resource("consensus").head(3)
    >>> map_df = pd.DataFrame(
    ...     {"source": ["LGALS9", "PTPRC", "MET", "CD44"], "target": ["Lgals9", "Ptprc", "Met", "Cd44"]}
    ... )
    >>> li.rs.translate_resource(resource, map_df)
       ligand receptor
    0  Lgals9    Ptprc
    1  Lgals9      Met
    2  Lgals9     Cd44
    """
    if columns is None:
        columns = ["ligand", "receptor"]

    for column in columns:
        resource = translate_column(resource, map_df, column, replace=replace, one_to_many=one_to_many)

    return resource


def get_hcop_orthologs(
    target_organism: str = "mouse",
    url: str | None = None,
    filename: str | Path | None = None,
    min_evidence: int = 3,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download the HCOP orthology file and filter it by minimum evidence.

    Parameters
    ----------
    target_organism : str
        Target organism for orthology mapping. Default is ``"mouse"``.
        Supported values: ``anole_lizard``, ``c.elegans``, ``cat``, ``cattle``,
        ``chicken``, ``chimpanzee``, ``dog``, ``fruitfly``, ``horse``, ``macaque``,
        ``mouse``, ``opossum``, ``pig``, ``platypus``, ``rat``, ``s.cerevisiae``,
        ``s.pombe``, ``xenopus``, ``zebrafish``.
        The target-organism column in the returned DataFrame follows the pattern
        ``{target_organism}_symbol`` (e.g. ``mouse_symbol``, ``rat_symbol``).
    url : str, optional
        Override the download URL. If ``None`` (default), the URL is constructed
        from ``target_organism`` using the HGNC Google Cloud Storage bucket.
    filename : str, optional
        Local filename to save the downloaded file. Derived from the URL if ``None``.
    min_evidence : int
        Minimum number of orthology resources that must support an interaction.
    columns : list, optional
        Columns to keep in the final DataFrame. If ``None``, all columns are kept.

    Returns
    -------
    mapping
        DataFrame with the HCOP mapping.

    Notes
    -----
    HCOP is a composite database combining data from various orthology resources.
    It provides a comprehensive set of human orthologs across many species.

    If you use this function, please reference the original HCOP papers:
    - Eyre, T.A., Wright, M.W., Lush, M.J. and Bruford, E.A., 2007. HCOP: a searchable database of human orthology predictions. Briefings in bioinformatics, 8(1), pp.2-5.
    - Yates, B., Gray, K.A., Jones, T.E. and Bruford, E.A., 2021. Updates to HCOP: the HGNC comparison of orthology predictions tool. Briefings in Bioinformatics, 22(6), p.bbab155.

    For more information, please visit the HCOP website: https://www.genenames.org/tools/hcop/

    Examples
    --------
    This function downloads from HCOP, so it is not run here. A typical call
    keeps only the two symbol columns needed by
    :func:`liana.rs.translate_resource`::

        map_df = get_hcop_orthologs(
            target_organism="mouse",
            columns=["human_symbol", "mouse_symbol"],
            min_evidence=3,
        ).rename(columns={"human_symbol": "source", "mouse_symbol": "target"})
    """
    if url is None:
        url = f"{_HCOP_BASE}/human_{target_organism}_hcop_fifteen_column.txt.gz"
    # check if exists
    if filename is None:
        path = Path(pooch.retrieve(url, known_hash=None, fname=url.rsplit("/", 1)[-1], path=sc.settings.datasetdir))
    else:
        path = Path(filename)
        if not path.exists():
            pooch.retrieve(url, known_hash=None, fname=path.name, path=path.parent)

    mapping = pd.read_csv(path, sep="\t")
    mapping["evidence"] = mapping["support"].apply(lambda x: len(x.split(",")))
    mapping = mapping[mapping["evidence"] >= min_evidence]

    if columns is not None:
        mapping = mapping[columns]

    return mapping
