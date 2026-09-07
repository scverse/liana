from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._types import RowFilter, get_obs
from liana.plotting._common import _filter_by, _get_top_n, _invert_scores, _prep_liana_res

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _pivot_liana_res(
    liana_res: pd.DataFrame,
    source_key: str = "source",
    target_key: str = "target",
    score_key: str | None = "lr_means",
    mode: Literal["counts", "mean"] = "counts",
) -> pd.DataFrame:
    if mode == "counts":
        pivot_table = liana_res.pivot_table(index=source_key, columns=target_key, aggfunc="size", fill_value=0)
    elif mode == "mean":
        if score_key is None:
            raise ValueError("`score_key` must be provided when `mode='mean'`.")
        if score_key is None:
            raise ValueError("`score_key` must be provided!")
        pivot_table = liana_res.pivot_table(
            index=source_key, columns=target_key, values=score_key, aggfunc="mean", fill_value=0
        )

    return pivot_table


def _scale_list(arr: ArrayLike, min_val: float = 1, max_val: float = 5) -> NDArray[np.floating]:
    values = np.asarray(arr, dtype=float)
    arr_min = np.min(values)
    arr_max = np.max(values)

    scaled: NDArray[np.floating] = (values - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
    return scaled


def _set_adata_color(
    adata: AnnData,
    label: str,
    color_dict: dict[str, str] | None = None,
    hex: bool = True,
) -> AnnData:
    obs = get_obs(adata)
    obs[label] = obs[label].astype("category")
    if color_dict:
        if not hex:
            from matplotlib.colors import to_hex

            color_dict = {x: to_hex(y) for x, y in color_dict.items()}

        _dt = _get_adata_colors(adata, label)
        _dt.update(color_dict)
        color_dict = _dt
        adata.uns[f"{label}_colors"] = [color_dict[x] for x in obs[label].cat.categories]
    else:
        if f"{label}_colors" not in adata.uns:
            # Handle both old (_set_default...) and new (set_default...) scanpy API
            for candidate in ("_set_default_colors_for_categorical_obs", "set_default_colors_for_categorical_obs"):
                _set_colors = getattr(sc.pl._utils, candidate, None)
                if _set_colors is not None:
                    _set_colors(adata, label)
                    break
            else:
                raise AttributeError(
                    "scanpy provides neither `_set_default_colors_for_categorical_obs` "
                    "nor `set_default_colors_for_categorical_obs`."
                )

    return adata


def _get_adata_colors(adata: AnnData, label: str) -> dict[str, str]:
    if f"{label}_colors" not in adata.uns:
        _set_adata_color(adata, label)
    return dict(zip(get_obs(adata)[label].cat.categories, adata.uns[f"{label}_colors"], strict=False))


def get_mask_df(
    pivot_table: pd.DataFrame,
    source_cell_type: list[str] | str | None = None,
    target_cell_type: list[str] | str | None = None,
    mode: Literal["and", "or"] = "or",
) -> pd.DataFrame:
    """
    Applies masking to a given table based on given source and/or target cell types and mode.

    Parameters
    ----------
    pivot_table
        The table to apply masking to.
    source_cell_type
        Source cell type(s) to mask for.
    target_cell_type
        Target cell type(s) to mask for.
    mode
        Which way to apply the mask, can be either `'and'` or `'or'` (i.e. intersection or union).

    Returns
    -------
    The resulting masked table
    """
    if source_cell_type is None and target_cell_type is None:
        return pivot_table

    if isinstance(source_cell_type, str):
        source_cell_type = [source_cell_type]
    if isinstance(target_cell_type, str):
        target_cell_type = [target_cell_type]

    mask_df = pd.DataFrame(np.zeros_like(pivot_table), index=pivot_table.index, columns=pivot_table.columns, dtype=bool)

    if mode == "or":
        if source_cell_type is not None:
            mask_df.loc[source_cell_type] = True
        if target_cell_type is not None:
            mask_df.loc[:, target_cell_type] = True
    elif mode == "and":
        if source_cell_type is not None and target_cell_type is not None:
            mask_df.loc[source_cell_type, target_cell_type] = True

    return pivot_table[mask_df].fillna(0)


@d.dedent
def circle(
    adata: sc.AnnData,
    uns_key: str = K.uns_key,
    liana_res: pd.DataFrame | None = None,
    groupby: str | None = None,
    source_key: str = "source",
    target_key: str = "target",
    score_key: str | None = None,
    inverse_score: bool = False,
    top_n: int | None = None,
    orderby: str | None = None,
    orderby_ascending: bool | None = None,
    orderby_absolute: bool = False,
    filter_fn: RowFilter | None = None,
    source_labels: list[str] | str | None = None,
    target_labels: list[str] | str | None = None,
    ligand_complex: list[str] | str | None = None,
    receptor_complex: list[str] | str | None = None,
    pivot_mode: Literal["counts", "mean"] = "counts",
    mask_mode: Literal["and", "or"] = "or",
    figure_size: tuple[float, float] = (5, 5),
    edge_alpha: float = 0.5,
    edge_arrow_size: int = 10,
    edge_width_scale: tuple[float, float] = (1, 5),
    node_alpha: float = 1,
    node_size_scale: tuple[float, float] = (100, 400),
    node_label_offset: tuple[float, float] = (0.1, -0.2),
    node_label_size: int = 8,
    node_label_alpha: float = 0.7,
) -> Axes:
    """
    Visualize the cell-cell communication network using a circular plot.

    Parameters
    ----------
    %(adata)s
    %(uns_key)s
    %(liana_res)s
    %(groupby)s
    %(source_key)s
    %(target_key)s
    %(score_key)s
    inverse_score
        Whether to invert the score, by default False. If True, the score will be -log10(score).
    %(top_n)s
    %(orderby)s
    %(orderby_ascending)s
    %(orderby_absolute)s
    %(filter_fn)s
    %(source_labels)s
    %(target_labels)s
    %(ligand_complex)s
    %(receptor_complex)s
    pivot_mode
        The mode of the pivot table, by default 'counts'.
        - 'counts': The number of connections between source and target.
        - 'mean': The mean of the values of `score_key` between source and target cell types (groupby).
        Note that `filter_fn` differs by pivot_mode: when counts it would remove all interactions
        that don't pass the filter, while for 'mean' it would retain interactions don't pass the filter
        if the same interaction passes it for any cell type pair.
    mask_mode
        The mode of the mask, by default 'or'.
        - 'or': Include the source or target cell type.
        - 'and': Include the source and target cell type.
    %(figure_size)s
    edge_alpha
        The transparency of the edges, by default .5.
    edge_arrow_size
        The size of the arrow, by default 10.
    edge_width_scale
        The scale of the edge width, by default (1, 5).
    node_alpha
        The transparency of the nodes, by default 1.
    node_size_scale
        The scale of the node size, by default (100, 400).
    node_label_offset
        The offset of the node label, by default (0.1, -0.2).
    node_label_size
        The size of the node label, by default 8.
    node_label_alpha
        The transparency of the node label, by default .7.

    Returns
    -------
    The figure axes containing the circle plot.

    Raises
    ------
    ValueError
        If `groupby` is not provided

    Examples
    --------
    `groupby` must name the `.obs` column that the interactions' `source`/`target`
    labels came from, so that its colours can be re-used for the nodes:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_adata()
    >>> li.mt.rank_aggregate(adata, groupby="bulk_labels", n_perms=None)
    >>> ax = li.pl.circle(adata, groupby="bulk_labels")

    With `pivot_mode='mean'` and a `score_key` the edges are weighted by that
    score's mean instead of by the number of interactions.
    """
    if groupby is None:
        raise ValueError("`groupby` must be provided!")

    liana_res = _prep_liana_res(
        adata=adata,
        liana_res=liana_res,
        source_labels=None,
        target_labels=None,
        ligand_complex=ligand_complex,
        receptor_complex=receptor_complex,
        uns_key=uns_key,
    )

    if pivot_mode == "counts":
        if filter_fn is not None:
            mask = liana_res.apply(filter_fn, axis=1).astype(bool)
            liana_res = liana_res[mask]
    elif pivot_mode == "mean":
        liana_res = _filter_by(liana_res, filter_fn)
    else:
        raise ValueError("`pivot_mode` must be 'counts' or 'mean'!")
    liana_res = _get_top_n(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)

    if inverse_score:
        liana_res[score_key] = _invert_scores(liana_res[score_key])

    pivot_table = _pivot_liana_res(
        liana_res, source_key=source_key, target_key=target_key, score_key=score_key, mode=pivot_mode
    )

    groupby_colors = _get_adata_colors(adata, label=groupby)

    # Mask pivot table
    _pivot_table = get_mask_df(
        pivot_table, source_cell_type=source_labels, target_cell_type=target_labels, mode=mask_mode
    )

    G = nx.from_pandas_adjacency(_pivot_table, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)

    edge_color = [groupby_colors[cell[0]] for cell in G.edges]
    edge_width = np.asarray([G.edges[e]["weight"] for e in G.edges()])
    edge_width = _scale_list(edge_width, max_val=edge_width_scale[1], min_val=edge_width_scale[0])

    node_color = [groupby_colors[cell] for cell in G.nodes]
    node_size = pivot_table.sum(axis=1).to_numpy()
    node_size = _scale_list(node_size, max_val=node_size_scale[1], min_val=node_size_scale[0])

    fig, ax = plt.subplots(figsize=figure_size)

    # Visualize network
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=edge_alpha,
        arrowsize=edge_arrow_size,
        arrowstyle="-|>",
        width=edge_width,
        edge_color=edge_color,
        connectionstyle="arc3,rad=-0.3",
        ax=ax,
    )

    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, alpha=node_alpha, ax=ax)
    label_options = {"ec": "k", "fc": "white", "alpha": node_label_alpha}
    _ = nx.draw_networkx_labels(
        G,
        {k: v + np.array(node_label_offset) for k, v in pos.items()},
        font_size=node_label_size,
        bbox=label_options,
        ax=ax,
    )

    ax.set_frame_on(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeff = 1.2
    ax.set_xlim((xlim[0] * coeff, xlim[1] * coeff))
    ax.set_ylim((ylim[0] * coeff, ylim[1]))
    ax.set_aspect("equal")

    return ax
