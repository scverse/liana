from __future__ import annotations

from anndata import AnnData
from pandas import DataFrame
from plotnine import (
    aes,
    element_rect,
    element_text,
    facet_grid,
    geom_point,
    ggplot,
    labs,
    scale_color_cmap,
    scale_size_continuous,
    theme,
    theme_bw,
)

from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._types import RowFilter
from liana.plotting._common import _check_var, _filter_by, _get_top_n, _invert_scores, _prep_liana_res


@d.dedent
def dotplot(
    adata: AnnData | None = None,
    uns_key: str = K.uns_key,
    liana_res: DataFrame | None = None,
    colour: str | None = None,
    size: str | None = None,
    source_labels: list[str] | None = None,
    target_labels: list[str] | None = None,
    top_n: int | None = None,
    orderby: str | None = None,
    orderby_ascending: bool | None = None,
    orderby_absolute: bool = False,
    filter_fn: RowFilter | None = None,
    ligand_complex: str | None = None,
    receptor_complex: str | None = None,
    inverse_colour: bool = False,
    inverse_size: bool = False,
    cmap: str = V.cmap,
    size_range: tuple[float, float] = (2, 9),
    figure_size: tuple[float, float] = (8, 6),
    return_fig: bool = V.return_fig,
) -> ggplot | None:
    """
    Dotplot interactions by source and target cells

    Parameters
    ----------
    %(adata)s
    %(uns_key)s
    %(liana_res)s
    %(colour)s
    %(size)s
    %(source_labels)s
    %(target_labels)s
    %(top_n)s
    %(orderby)s
    %(orderby_ascending)s
    %(orderby_absolute)s
    %(filter_fn)s
    %(ligand_complex)s
    %(receptor_complex)s
    %(inverse_colour)s
    %(inverse_size)s
    %(cmap)s
    %(size_range)s
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    The resulting dotplot

    Examples
    --------
    Pass either a method's result as `liana_res`, or the `adata` it was written to:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> li.mt.rank_aggregate(adata, groupby="bulk_labels", n_perms=None)
    >>> p = li.pl.dotplot(
    ...     adata,
    ...     colour="lr_means",
    ...     size="magnitude_rank",
    ...     inverse_size=True,
    ...     top_n=10,
    ...     orderby="magnitude_rank",
    ...     orderby_ascending=True,
    ... )
    """
    liana_res = _prep_liana_res(
        adata=adata,
        liana_res=liana_res,
        source_labels=source_labels,
        target_labels=target_labels,
        ligand_complex=ligand_complex,
        receptor_complex=receptor_complex,
        uns_key=uns_key,
    )
    colour = _check_var(liana_res, var=colour, var_name="colour")
    size = _check_var(liana_res, var=size, var_name="size")

    liana_res = _filter_by(liana_res, filter_fn)
    liana_res = _get_top_n(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)

    # inverse sc if needed
    if inverse_colour:
        liana_res[colour] = _invert_scores(liana_res[colour])
    if inverse_size:
        liana_res[size] = _invert_scores(liana_res[size])

    # generate plot
    p = (
        ggplot(liana_res, aes(x="target", y="interaction", colour=colour, size=size))
        + geom_point()
        + facet_grid("~source")
        + scale_size_continuous(range=size_range)
        + scale_color_cmap(cmap)
        + labs(
            color=str.capitalize(colour),
            size=str.capitalize(size),
            y="Interactions (Ligand -> Receptor)",
            x="Target",
            title="Source",
        )
        + theme_bw()
        + theme(
            legend_text=element_text(size=14),
            strip_background=element_rect(fill="white"),
            strip_text=element_text(size=15, colour="black"),
            axis_text_y=element_text(size=10, colour="black"),
            axis_title_y=element_text(colour="#808080", face="bold", size=15),
            axis_text_x=element_text(size=11, face="bold", angle=90),
            figure_size=figure_size,
            plot_title=element_text(vjust=0, hjust=0.5, face="bold", colour="#808080", size=15),
        )
    )

    if return_fig:
        return p

    p.draw()
    return None


@d.dedent
def dotplot_by_sample(
    adata: AnnData | None = None,
    uns_key: str = K.uns_key,
    liana_res: DataFrame | None = None,
    sample_key: str = "sample",
    colour: str | None = None,
    size: str | None = None,
    inverse_colour: bool = False,
    inverse_size: bool = False,
    source_labels: str | None = None,
    target_labels: str | None = None,
    ligand_complex: list[str] | str | None = None,
    receptor_complex: list[str] | str | None = None,
    size_range: tuple[float, float] = (2, 9),
    cmap: str = V.cmap,
    figure_size: tuple[float, float] = (8, 6),
    return_fig: bool = V.return_fig,
) -> ggplot | None:
    """
    A dotplot of interactions by sample

    Parameters
    ----------
    %(adata)s
    %(uns_key)s
    %(liana_res)s
    sample_key
        sample_key used to group different samples/contexts from `liana_res`. Defaults to 'sample'.
    %(colour)s
    %(size)s
    %(inverse_colour)s
    %(inverse_size)s
    %(source_labels)s
    %(target_labels)s
    %(ligand_complex)s
    %(receptor_complex)s
    %(size_range)s
    %(cmap)s
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    Returns a dot plot for the specified interactions by sample.

    Examples
    --------
    Expects a by-sample result, as written by any method's `.by_sample`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> li.mt.rank_aggregate.by_sample(adata, sample_key="sample", groupby="bulk_labels", n_perms=None)
    >>> p = li.pl.dotplot_by_sample(adata, colour="lr_means", size="magnitude_rank", ligand_complex="HLA-DRA")
    """
    liana_res = _prep_liana_res(
        adata=adata,
        liana_res=liana_res,
        source_labels=source_labels,
        target_labels=target_labels,
        ligand_complex=ligand_complex,
        receptor_complex=receptor_complex,
        uns_key=uns_key,
    )
    colour = _check_var(liana_res, var=colour, var_name="colour")
    size = _check_var(liana_res, var=size, var_name="size")

    # inverse sc if needed
    if inverse_colour:
        liana_res[colour] = _invert_scores(liana_res[colour])
    if inverse_size:
        liana_res[size] = _invert_scores(liana_res[size])

    p = (
        ggplot(liana_res, aes(x="target", y="source", colour=colour, size=size))
        + geom_point()
        + facet_grid(f"interaction~{sample_key}", space="free", scales="free")
        + scale_size_continuous(range=size_range)
        + scale_color_cmap(name=cmap)
        + labs(
            color=str.capitalize(colour),
            size=str.capitalize(size),
            y="Source",
            x="Target",
            title=sample_key.capitalize(),
        )
        + theme_bw()
        + theme(
            legend_text=element_text(size=14),
            strip_background=element_rect(fill="white"),
            strip_text=element_text(size=13, colour="black", angle=90),
            axis_text_y=element_text(size=10, colour="black"),
            axis_title_y=element_text(colour="#808080", face="bold", size=12),
            axis_text_x=element_text(size=11, face="bold", angle=90),
            axis_title_x=element_text(colour="#808080", face="bold", size=12),
            figure_size=figure_size,
            plot_title=element_text(vjust=0, hjust=0.5, face="bold", size=12),
        )
    )
    if return_fig:
        return p

    p.draw()
    return None
