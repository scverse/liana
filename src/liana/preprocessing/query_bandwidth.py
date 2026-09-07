import numpy as np
from pandas import DataFrame
from plotnine import aes, geom_line, geom_point, ggplot, theme, theme_bw, xlab, ylab
from sklearn.neighbors import BallTree


def query_bandwidth(
    coordinates: np.ndarray,
    start: int = 0,
    end: int = 500,
    interval_n: int = 50,
    reference: np.ndarray | None = None,
    figure_size: tuple[float, float] = (6, 4),
) -> tuple[ggplot, DataFrame]:
    """
    Query the bandwidth (maximum distance) at which the average number of neighbors is maximized.

    Parameters
    ----------
    coordinates
        Spatial coordinates of spots.
    start
        Starting bandwidth.
    end
        Ending bandwidth.
    interval_n
        Number of bandwidths to query. Used to generate a linearly spaced
        interval between `start` and `end`. Default is 50.
    reference
        Reference coordinates to query the neighbors from. Default is `None`,
        which will use `coordinates`.
    figure_size
        Size of the returned figure as a `(width, height)` tuple.

    Returns
    -------
    A `plotnine` plot and a `pandas` DataFrame with the following columns:
        - `bandwith`: the bandwidth (maximum distance) at which the average
        number of neighbors is maximized.
        - `neighbours`: the average number of neighbors at the specified
        bandwidth.

    Examples
    --------
    Helps choose the `bandwidth` for :func:`liana.pp.spatial_neighbors` by
    showing how many neighbours each candidate value admits:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> fig, df = li.pp.query_bandwidth(adata.obsm["spatial"], start=0, end=1000)
    """
    tree = BallTree(coordinates, metric="euclidean")
    df = DataFrame()
    interval = np.linspace(start, end, interval_n)

    if reference is None:
        _reference = coordinates
    else:
        _reference = reference

    for n in range(interval_n):
        max_distance = interval[n]
        df.loc[n, "bandwith"] = max_distance

        # query the neighbors within the specified distance
        num_neighbors = tree.query_radius(_reference, r=max_distance, count_only=True)

        # calculate the average number of neighbors
        avg_nn = np.ceil(np.median(num_neighbors))
        df.loc[n, "neighbours"] = avg_nn - 1

    p = (
        ggplot(df, aes(x="bandwith", y="neighbours"))
        + geom_line()
        + geom_point()
        + theme_bw(base_size=16)
        + xlab("Bandwidth")
        + ylab("Number of Neighbors")
        + theme(figure_size=figure_size)
    )

    return p, df
