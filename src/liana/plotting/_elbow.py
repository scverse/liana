from __future__ import annotations

import pandas as pd
from anndata import AnnData
from plotnine import aes, geom_line, geom_point, geom_vline, ggplot, labs, scale_x_continuous, theme, theme_bw

from liana._core._constants import DefaultValues as V
from liana._core._docs import d


@d.dedent
def elbow(
    adata: AnnData | None = None,
    errors: pd.DataFrame | None = None,
    rank: int | None = None,
    figure_size: tuple[float, float] = (5, 4),
    return_fig: bool = V.return_fig,
) -> ggplot | None:
    """
    Plot the NMF reconstruction error per rank, with the estimated elbow.

    Parameters
    ----------
    adata
        AnnData on which :func:`liana.ms.nmf` was run with `n_components=None`;
        reads `adata.uns['nmf_errors']` and `adata.uns['nmf_rank']`.
    errors
        Error curve as returned by :func:`liana.ms.estimate_elbow`; used when `adata` is None.
    rank
        Estimated rank to mark with a dashed line; used when `adata` is None.
    %(figure_size)s
    %(return_fig)s

    Returns
    -------
    The resulting elbow plot.

    Raises
    ------
    ValueError
        If no error curve is available, e.g. `nmf` was run with a fixed `n_components`.

    Examples
    --------
    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> lrdata = li.mt.bivariate(adata, resource_name="consensus", local_name="cosine", global_name=None, n_perms=None)
    >>> li.ms.nmf(lrdata, n_components=None, k_range=range(1, 5), random_state=0, max_iter=200)
    >>> p = li.pl.elbow(lrdata)

    """
    if adata is not None:
        errors, rank = adata.uns.get("nmf_errors"), adata.uns.get("nmf_rank")
    if errors is None:
        raise ValueError("No error curve found. Run `li.ms.nmf` with `n_components=None`, or pass `errors`.")

    p = (
        ggplot(errors, aes(x="k", y="error"))
        + geom_line()
        + geom_point()
        + theme_bw()
        + scale_x_continuous(breaks=errors["k"].to_list())
        + labs(x="Component number (k)", y="Reconstruction error")
        + theme(figure_size=figure_size)
    )
    if rank is not None:
        p = p + geom_vline(xintercept=rank, linetype="dashed", color="red")

    if return_fig:
        return p

    p.draw()
    return None
