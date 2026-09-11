from __future__ import annotations

import inspect
import re
from collections.abc import Callable

# `%(name)s`, the placeholder syntax every docstring in this package is written in.
_PLACEHOLDER = re.compile(r"%\((\w+)\)s")


class DocstringProcessor:
    """Substitutes ``%(name)s`` placeholders into the docstrings it decorates.

    This replaces `docrep.DocstringProcessor`, which is unmaintained and ships no type information -- an untyped decorator makes every function it decorates untyped in turn, which `mypy --strict` rejects.
    `scanpy` carries an equivalent helper of its own (`scanpy._utils._doc_params`) for the same reason.

    The placeholder syntax is kept as docrep spelled it, so the docstrings themselves are unchanged.

    Examples
    --------
    >>> d = DocstringProcessor(greeting="hello")
    >>> @d.dedent
    ... def f():
    ...     '''Says %(greeting)s.'''
    >>> f.__doc__
    'Says hello.'
    """

    def __init__(self, **params: str) -> None:
        self.params = params

    def dedent[T: Callable[..., object] | type](self, obj: T, /) -> T:
        """Dedent ``obj``'s docstring and substitute the registered parameters.

        Raises
        ------
        KeyError
            If the docstring references a placeholder that was never registered.
            (docrep only warned, which let typos survive into the rendered docs.)
        """
        if obj.__doc__ is None:
            return obj

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in self.params:
                raise KeyError(f"{obj.__qualname__}: unknown docstring placeholder %({name})s")
            return self.params[name]

        obj.__doc__ = _PLACEHOLDER.sub(replace, inspect.cleandoc(obj.__doc__))
        return obj


# Common docstrings
_adata = """\
adata
    Annotated data object."""

_mdata = """\
mdata
    MuData (multimodal) data object."""

_misty = """\
misty
    MistyData object with modelling results
"""

_groupby = """\
groupby
    Key to be used for grouping."""

_n_perms = """\
n_perms
    Number of permutations for the permutation test. If None, no p-values are computed."""

_seed = """\
seed
    Random seed for reproducibility."""

_resource = """\
resource
    A pandas dataframe with [`ligand`, `receptor`] columns.
    If provided will overrule the resource requested via `resource_name`"""

_interactions = """\
interactions
    List of tuples with ligand-receptor pairs `[(ligand, receptor), ...]` to be used for the analysis.
    If passed, it will overrule the resource requested via `resource` and `resource_name`."""

_resource_name = """\
resource_name
    Name of the resource to be used for ligand-receptor inference. See `li.rs.show_resources()` for available resources."""

_db_path = """\
db_path
    Path to the SQLite database file. If None, the database is fetched into `scanpy.settings.datasetdir` and reused from there."""

_sample_key = """\
sample_key
    key in `adata.obs` to use for grouping by sample or context."""

_key_added = """\
key_added
    Key under which the results will be stored in `adata.uns` if `inplace` is True."""

_use_raw = """\
use_raw
    Whether to use the `.raw` attribute of adata. Defaults to False (uses `.X`)."""

_layer = """\
layer
    Layer in anndata.AnnData.layers to use. If None, use anndata.AnnData.X."""

_mdata_kwargs = """\
mdata_kwargs
    Keyword arguments to be passed to `li.ms.mdata_to_anndata` if `adata` is an instance of `MuData`.
    If an AnnData object is passed, these arguments are ignored."""

_inplace = """\
inplace
    Whether to store results in place, or else to return them."""

_verbose = """\
verbose
    Verbosity flag. `False`, the default, hides progress and info messages but
    still warns about data-altering conditions; `True` shows both; `None`
    silences everything, warnings included."""

_lr_sep = """\
lr_sep
    Separator to use when joining ligand and receptor names into interactions.
"""


_source_labels = """\
source_labels
    List of labels to use as `source`, the rest are filtered out.
"""

_target_labels = """\
target_labels
    List of labels to use as `target`, the rest are filtered out.
"""


# Single-cell specific docstrings
_n_perms_sc = """\
n_perms
    Number of permutations for the permutation test. Relevant only for permutation-based methods
    (e.g., `CellPhoneDB`). If `None` is passed, no permutation testing is performed."""

_expr_prop = """\
expr_prop
    Minimum expression proportion for the ligands and receptors (+ their subunits) within each
    cell-type group. Set to 0 to return unfiltered results."""

_nz_prop = """\
nz_prop
    Minimum proportion of all observations (cells/spots) with a non-zero value for a variable to be
    kept. Unlike `expr_prop`, it is computed across all observations, irrespective of any grouping.
    Set to 0 to return unfiltered results."""

_min_cells = """\
min_cells
    Minimum cells (per cell identity if grouped by `groupby`) to be considered for downstream analysis."""

_base = """\
base
    Exponent base used to reverse the log-transformation of the matrix. Relevant only for the `logfc` method."""

_return_all_lrs = """\
return_all_lrs
    Bool whether to return all ligand-receptor pairs, or only those that surpass the `expr_prop`
    threshold. Ligand-receptor pairs that do not pass the `expr_prop` threshold will be assigned
    to the *worst* score of the ones that do. `False` by default."""

_supp_columns = """\
supp_columns
    Additional columns to be added from any of the methods implemented in liana,
    or any of the columns returned by `scanpy.tl.rank_genes_groups`, each starting with ligand_* or receptor_*.
    For example, `['ligand_pvals', 'receptor_pvals']`. None by default."""

_de_method = """\
de_method
    Differential expression method. `scanpy.tl.rank_genes_groups` is used to rank genes
    according to 1vsRest. The default method is 't-test'."""

_groupby_pairs = """\
groupby_pairs
    A DataFrame with columns `source` and `target` to be used to subset the possible combinations of interacting cell types.
    If None, all possible combinations are used."""


# multi-condition specific docstrings
_source_key = """\
source_key
    Column name of the sender/source cell types in `liana_res`."""

_target_key = """\
target_key
    Column name of the receiver/target cell types in `liana_res`."""

_ligand_key = """\
ligand_key
    Column name of the ligand in `liana_res`."""

_receptor_key = """\
receptor_key
    Column name of the receptor in `liana_res`."""

_score_key = """\
score_key
    Column name of the score in `liana_res`. If None, the score is inferred from the method."""

_uns_key = """\
uns_key
    Key in `adata.uns` that contains the LIANA results. Default is `'liana_res'`."""


_inverse_fn = """\
inverse_fn
    Function applied to scores for which a *lower* value is the stronger one -- p-values
    and aggregate ranks such as `magnitude_rank` -- so that "higher is stronger" holds
    throughout. Defaults to `-log10(x + eps)`. Which scores are inverted is decided by
    `liana.mt.get_method_scores`, so this is handled automatically for liana's own scores."""


# Spatial specific docstrings
_spatial_key = """\
spatial_key
    Key in `adata.obsm` that contains the spatial coordinates."""

_connectivity_key = """\
connectivity_key
    Key in `adata.obsp` that contains the spatial connectivity matrix. Default is `'spatial_connectivities'`. """

_local_name = """\
local_name
    Name of the local function to use for the analysis. Passing `None` will return only the Global scores."""

_global_name = """\
global_name
    Name or names (list) of the global function(s) to use for the analysis. Passing `None` will not calculate any global scores"""

_positive_only = """\
positive_only
    Whether to mask non-positive interactions."""

_add_categories = """\
add_categories
    Whether to add categories about the local scores."""

_x_name = """\
x_name
    Name of the x-variable. If passing a `resource` dataframe, this should match the first column."""

_y_name = """\
y_name
    Name of the y-variable. If passing a `resource` dataframe, this should match the second column."""

_mask_negatives = """\
mask_negatives
    Whether to mask negative-negative (low-low) or uncategorized interactions."""

_spatial_kwargs = """
spatial_kwargs
    Keyword arguments passed to `liana.pp.spatial_pair_proximity()` for computing
    spatial proximity weights. Default is None, which uses default values
    (bandwidth=250, kernel='gaussian', trim_fraction=0.1)."""

_kernel = """
kernel
    Kernel function used to generate connectivity/proximity weights.
    It controls the shape of the connectivity weights.
    The following options are available: ['gaussian', 'exponential', 'linear', 'misty_rbf']."""

_coordinates = """
coordinates
    Spatial coordinates (e.g., in µm) with shape (n_cells, n_dimensions)."""

_bandwidth = """
bandwidth
    Denotes signaling length and controls the maximum distance at which two spots/cells are considered.
    Corresponds to the units in which spatial coordinates are expressed."""

_contact_bandwidth = """
contact_bandwidth
    Bandwidth for contact proximity calculation and distance threshold for contact interactions.
    If None, contact proximity is not calculated. Default is None."""

_max_radius = """\
max_radius
    Inner edge of the last (widest) annulus bin; the outer edge extends
    to `max_radius + annulus_steps * radius_step`."""

_radius_step = """\
radius_step
    Step between successive annulus inner edges."""

_annulus_steps = """\
annulus_steps
    Ring width of each annulus, in units of `radius_step`
    (width = `annulus_steps * radius_step`). `1` (default) gives
    disjoint annuli that tile the range without gaps or overlap;
    `> 1` gives overlapping annuli -- a `annulus_steps`-wide moving
    window over the same tiles, i.e. a smoothed `g(r)`."""

_extend_first_annulus = """\
extend_first_annulus
    If `True` (default), extend the first annulus inward to start at
    radius 0 (spanning `[0, (1 + annulus_steps) * radius_step)`) rather
    than at `radius_step`. Cell centroids cannot lie closer than ~one cell
    diameter, so the innermost band is otherwise a thin, near-empty,
    high-variance bin; extending it folds genuine cell-cell contact
    pairs into the first bin instead of discarding them. `False` keeps
    the first annulus at `[radius_step, (1 + annulus_steps) * radius_step)`."""

_cell_types = """\
cell_types
    Subset of cell types to consider. Defaults to all types in
    `adata.obs[groupby]`."""


# Plot docstrings
_liana_res = """\
liana_res
    `liana_res` a `DataFrame` in liana's format.
"""

_colour = """\
colour
    `column` in `liana_res` to define the colours of the dots.
"""

_size = """\
size
    `column` in `liana_res` to define the size of the dots.
"""

_top_n = """\
top_n
    ``top_n`` entities to plot.
"""

_orderby = """\
orderby
    If `top_n` is not `None`, order the interactions by this column
"""

_orderby_ascending = """\
orderby_ascending
    If `top_n` is not `None`, specify how to order the interactions
"""

_orderby_absolute = """\
orderby_absolute
    If `top_n` is not `None`, whether to order by the absolute value of the `orderby` column.
"""

_filter_fn = """\
filter_fn
    A function, applied along the columns (axis=1), used to filter the results to be plotted.
"""

_aggregate_fn = """\
aggregate_fn
    A function used to aggregate the results to be plotted.
"""

_ligand_complex = """\
ligand_complex
    `list` of ligand complexes to filter the interactions to be plotted. Defaults to None.
"""

_receptor_complex = """\
receptor_complex
    `list` of receptor complexes to filter the interactions to be plotted. Defaults to None.
"""


_inverse_colour = """\
inverse_colour
    Whether to -log10 the `colour` column for plotting. `False` by default.
"""

_inverse_size = """\
inverse_size
    Whether to -log10 the `size` column for plotting. `False` by default.
"""

_size_range = """\
size_range
    Define size range. Tuple of (min, max) integers.
"""

_cmap = """\
cmap
    Colour map to use for plotting.
"""

_figure_size = """\
figure_size
    Figure x,y size
"""

_return_fig = """\
return_fig
    `bool` whether to return the fig object.
"""


d = DocstringProcessor(
    adata=_adata,
    mdata=_mdata,
    misty=_misty,
    groupby=_groupby,
    groupby_pairs=_groupby_pairs,
    seed=_seed,
    resource=_resource,
    interactions=_interactions,
    resource_name=_resource_name,
    db_path=_db_path,
    sample_key=_sample_key,
    key_added=_key_added,
    use_raw=_use_raw,
    layer=_layer,
    mdata_kwargs=_mdata_kwargs,
    inplace=_inplace,
    verbose=_verbose,
    lr_sep=_lr_sep,
    n_perms=_n_perms,
    n_perms_sc=_n_perms_sc,
    expr_prop=_expr_prop,
    nz_prop=_nz_prop,
    min_cells=_min_cells,
    base=_base,
    return_all_lrs=_return_all_lrs,
    supp_columns=_supp_columns,
    liana_res=_liana_res,
    de_method=_de_method,
    source_key=_source_key,
    target_key=_target_key,
    ligand_key=_ligand_key,
    receptor_key=_receptor_key,
    score_key=_score_key,
    uns_key=_uns_key,
    inverse_fn=_inverse_fn,
    spatial_key=_spatial_key,
    connectivity_key=_connectivity_key,
    local_name=_local_name,
    global_name=_global_name,
    positive_only=_positive_only,
    add_categories=_add_categories,
    x_name=_x_name,
    y_name=_y_name,
    mask_negatives=_mask_negatives,
    colour=_colour,
    size=_size,
    source_labels=_source_labels,
    target_labels=_target_labels,
    top_n=_top_n,
    orderby=_orderby,
    orderby_ascending=_orderby_ascending,
    orderby_absolute=_orderby_absolute,
    filter_fn=_filter_fn,
    aggregate_fn=_aggregate_fn,
    ligand_complex=_ligand_complex,
    receptor_complex=_receptor_complex,
    inverse_colour=_inverse_colour,
    inverse_size=_inverse_size,
    size_range=_size_range,
    cmap=_cmap,
    figure_size=_figure_size,
    return_fig=_return_fig,
    spatial_kwargs=_spatial_kwargs,
    bandwidth=_bandwidth,
    contact_bandwidth=_contact_bandwidth,
    kernel=_kernel,
    coordinates=_coordinates,
    max_radius=_max_radius,
    radius_step=_radius_step,
    annulus_steps=_annulus_steps,
    extend_first_annulus=_extend_first_annulus,
    cell_types=_cell_types,
)
