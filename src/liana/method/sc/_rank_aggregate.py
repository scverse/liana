from __future__ import annotations

from typing import Literal

import anndata as an
from mudata import MuData
from pandas import DataFrame

from liana._core._constants import DefaultValues as V
from liana._core._constants import DeMethod
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana.method.sc._liana_pipe import MdataKwargs, SpatialKwargs, liana_pipe_consensus
from liana.method.sc._Method import Method, MethodMeta


class AggregateClass(MethodMeta):
    """
    LIANA's Method Consensus Class

    Parameters
    ----------
    _SCORE
        Scoring method instance
    methods
        List of method instances to aggregate

    Attributes
    ----------
    _SCORE
        Scoring method instance
    methods
        List of method instances to aggregate
    specificity_specs
        Collection of method specificity's specifications
    magnitude_specs
        Collection of method magnitudes' specifications
    add_cols
        Additional columns required for each method
    complex_cols
        Columns relevant for protein complexes for each method
    """

    def __init__(self, _SCORE: MethodMeta, methods: list[Method]) -> None:
        super().__init__(
            method_name=_SCORE.method_name,
            complex_cols=[],
            add_cols=[],
            fun=_SCORE.fun,
            magnitude=_SCORE.magnitude,
            magnitude_ascending=True,
            specificity=_SCORE.specificity,
            specificity_ascending=True,
            permute=_SCORE.permute,
            reference=_SCORE.reference,
        )
        self._SCORE = _SCORE
        self.methods = methods

        # Define sc to aggregate
        self.specificity_specs: dict[str, tuple[str, bool | None]] = {
            method.method_name: (method.specificity, method.specificity_ascending)
            for method in methods
            if method.specificity is not None
        }
        self.magnitude_specs = {
            method.method_name: (method.magnitude, method.magnitude_ascending)
            for method in methods
            if method.magnitude is not None
        }

        # Define additional columns needed depending on the methods to be run
        self.add_cols = list({x for li in [method.add_cols for method in methods] for x in li})
        self.complex_cols = list({x for li in [method.complex_cols for method in methods] for x in li})

    def describe(self) -> None:
        """Briefly describes the method"""
        print(
            f"{self.method_name} returns `{self.magnitude}`, `{self.specificity}`. "
            f"{self.magnitude} and {self.specificity} respectively represent an aggregate of the "
            f"`magnitude`- and `specificity`-related scoring functions from the different methods."
        )

    @d.dedent
    def __call__(
        self,
        adata: an.AnnData | MuData,
        groupby: str,
        resource_name: str = V.resource_name,
        expr_prop: float = V.expr_prop,
        min_cells: int = V.min_cells,
        groupby_pairs: DataFrame | None = V.groupby_pairs,
        base: float = V.logbase,
        supp_columns: list[str] | None = V.supp_columns,
        aggregate_method: Literal["rra", "mean"] = "rra",
        consensus_opts: list[str] | Literal[False] | None = None,
        return_all_lrs: bool = V.return_all_lrs,
        key_added: str = K.uns_key,
        use_raw: bool = V.use_raw,
        layer: str | None = V.layer,
        de_method: DeMethod = V.de_method,
        n_perms: int | None = V.n_perms,
        seed: int = V.seed,
        n_jobs: int = 1,
        resource: DataFrame | None = V.resource,
        interactions: list[tuple[str, str]] | None = V.interactions,
        mdata_kwargs: MdataKwargs | None = None,
        spatial_key: str | None = None,
        spatial_kwargs: SpatialKwargs | None = None,
        inplace: bool = V.inplace,
        verbose: bool | None = V.verbose,
    ) -> DataFrame | dict[str, DataFrame] | None:
        """
        Get an aggregate of ligand-receptor scores from multiple methods.

        Parameters
        ----------
        %(adata)s
        %(groupby)s
        %(resource_name)s
        %(expr_prop)s
        %(min_cells)s
        %(groupby_pairs)s
        %(base)s
        %(supp_columns)s
        aggregate_method
            Method aggregation approach, one of ['mean', 'rra'], where `mean` represents the
            mean rank, while 'rra' is the RobustRankAggregate (Kolde et al., 2014)
            of the interactions
        consensus_opts
            Strategies to aggregate interactions across methods.
            Default is None - i.e. ['Specificity', 'Magnitude'] and both specificity and magnitude are aggregated.
            Methods whose score was not computed (e.g. permutation p-values when `n_perms` is None) are skipped.
        %(return_all_lrs)s
        %(key_added)s
        %(use_raw)s
        %(layer)s
        %(de_method)s
        %(n_perms)s
        %(seed)s
        n_jobs
            Number of jobs to run in parallel.
        %(resource)s
        %(interactions)s
        %(mdata_kwargs)s
        %(spatial_key)s
        %(spatial_kwargs)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        If ``inplace = False``, returns a `DataFrame` with ligand-receptor results
        Otherwise, modifies the ``adata`` object with the following key:

            - :attr:`anndata.AnnData.uns` ``['liana_res']`` with the aforementioned DataFrame

        Examples
        --------
        Runs the individual methods and aggregates their ranks. `n_perms=None` skips
        the permutations that the permutation-based methods would otherwise need --
        drop it, or set an integer, for a real analysis:

        >>> import liana as li
        >>> adata = li.ds.generate_toy_adata()
        >>> li.mt.rank_aggregate(adata, groupby="bulk_labels", n_perms=None)

        The frame carries `magnitude_rank` and `specificity_rank` alongside each aggregated method's own scores -- ``li.mt.rank_aggregate.describe()`` says what the ranks mean, and ``liana.mt.rank_aggregate`` which methods go into them.
        """
        if mdata_kwargs is None:
            mdata_kwargs = {}
        liana_res = liana_pipe_consensus(
            adata=adata,
            groupby=groupby,
            resource_name=resource_name,
            resource=resource,
            groupby_pairs=groupby_pairs,
            interactions=interactions,
            expr_prop=expr_prop,
            min_cells=min_cells,
            base=base,
            supp_columns=supp_columns,
            return_all_lrs=return_all_lrs,
            de_method=de_method,
            verbose=verbose,
            consensus=self,
            use_raw=use_raw,
            layer=layer,
            n_perms=n_perms,
            seed=seed,
            n_jobs=n_jobs,
            aggregate_method=aggregate_method,
            consensus_opts=consensus_opts,
            spatial_key=spatial_key,
            spatial_kwargs=spatial_kwargs,
            mdata_kwargs=mdata_kwargs,
        )

        if inplace:
            adata.uns[key_added] = liana_res
        return None if inplace else liana_res


_rank_aggregate_meta = MethodMeta(
    method_name="Rank_Aggregate",
    complex_cols=[],
    add_cols=[],
    fun=None,  # change to _robust_rank
    magnitude="magnitude_rank",
    magnitude_ascending=True,
    specificity="specificity_rank",
    specificity_ascending=True,
    permute=False,
    reference="Dimitrov, D., Türei, D., Garrido-Rodriguez, M., Burmedi, P.L., "
    "Nagai, J.S., Boys, C., Ramirez Flores, R.O., Kim, H., Szalai, B., "
    "Costa, I.G. and Valdeolivas, A., 2022. Comparison of methods and "
    "resources for cell-cell communication inference from single-cell "
    "RNA-Seq data. Nature Communications, 13(1), pp.1-13. ",
)
