from __future__ import annotations

from collections.abc import Callable

import anndata as ad
import pandas as pd
import plotnine as p9

from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._types import RowFilter
from liana.plotting._common import _check_var, _filter_by, _get_top_n, _prep_liana_res


@d.dedent
def tileplot(
    adata: ad.AnnData | None = None,
    liana_res: pd.DataFrame | None = None,
    fill: str | None = None,
    label: str | None = None,
    label_fn: Callable[[object], str] | None = None,
    source_labels: str | list[str] | None = None,
    target_labels: str | list[str] | None = None,
    ligand_complex: str | list[str] | None = None,
    receptor_complex: str | list[str] | None = None,
    uns_key: str = K.uns_key,
    top_n: int | None = None,
    orderby: str | None = None,
    orderby_ascending: bool = False,
    orderby_absolute: bool = True,
    filter_fn: RowFilter | None = None,
    source_title: str | None = None,
    target_title: str | None = None,
    cmap: str = V.cmap,
    figure_size: tuple[float, float] = (5, 5),
    label_size: int = 12,
    return_fig: bool = V.return_fig,
) -> p9.ggplot | None:
    """
    Tileplot interactions by source and target cells

    Parameters
    ----------
    %(adata)s
    %(liana_res)s
    fill
        `column` in `liana_res` to define the fill of the tiles
    label
        `column` in `liana_res` to define the label of the tiles
    label_fn
        `callable` to apply to the `label` column
    %(source_labels)s
    %(target_labels)s
    %(ligand_complex)s
    %(receptor_complex)s
    %(uns_key)s
    %(top_n)s
    %(orderby)s
    %(orderby_ascending)s
    %(orderby_absolute)s
    %(filter_fn)s
    source_title
        Title for the source facet. Default is 'Source'
    target_title
        Title for the target facet. Default is 'Target'
    %(cmap)s
    %(figure_size)s
    label_size
        Size of the label text
    %(return_fig)s

    Returns
    -------
    The resulting tileplot

    Examples
    --------
    `fill` and `label` each name a suffix shared by a `ligand_`/`receptor_` pair of
    columns, so both sides of every interaction are tiled next to each other:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> li.mt.cellphonedb(adata, groupby="bulk_labels", n_perms=100)
    >>> p = li.pl.tileplot(adata, fill="means", label="props", top_n=10, orderby="lr_means")
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

    liana_res = _filter_by(liana_res, filter_fn)
    liana_res = _get_top_n(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)

    # checked here because `_check_var` below runs after `fill`/`label` are used
    if fill is None:
        raise ValueError("`fill` must be provided!")
    if label is None:
        raise ValueError("`label` must be provided!")

    # get columns which ends with fill or label
    relevant_cols = [col for col in liana_res.columns if col.endswith(fill) | col.endswith(label)]

    ligand_stats = _entity_stats(
        liana_res, entity="ligand", entity_type="source", relevant_cols=relevant_cols, type_title=source_title
    )

    _check_var(ligand_stats, var=fill, var_name="fill")
    _check_var(ligand_stats, var=label, var_name="label")

    receptor_stats = _entity_stats(
        liana_res, entity="receptor", entity_type="target", relevant_cols=relevant_cols, type_title=target_title
    )

    liana_res = pd.concat([ligand_stats, receptor_stats])

    if label_fn is not None:
        liana_res[label] = liana_res[label].apply(label_fn)

    p = (
        p9.ggplot(liana_res, p9.aes(x="cell_type", y="interaction", fill=fill))
        + p9.geom_tile()
        + p9.geom_text(p9.aes(label=label), size=label_size, color="white")
        + p9.facet_grid("~ type", scales="free")
        + p9.theme_bw(base_size=14)
        + p9.theme(
            axis_text_x=p9.element_text(angle=90),
            figure_size=figure_size,
            strip_background=p9.element_rect(colour="black", fill="#fdfff4"),
        )
        + p9.scale_fill_cmap(cmap)
        + p9.labs(x="Cell type", y="Interaction", fill=str.capitalize(fill))
    )

    if return_fig:
        return p

    p.draw()
    return None


def _entity_stats(
    liana_res: pd.DataFrame,
    entity: str,
    entity_type: str,
    relevant_cols: list[str],
    type_title: str | None = None,
) -> pd.DataFrame:
    entity_stats = liana_res[["interaction", f"{entity}_complex", entity_type, *relevant_cols]].copy()
    if type_title is None:
        type_title = entity_type.capitalize()
    entity_stats = entity_stats.rename(columns={entity_type: "cell_type"}).assign(type=type_title)
    entity_stats.columns = entity_stats.columns.str.replace(entity + "_", "")
    return entity_stats
