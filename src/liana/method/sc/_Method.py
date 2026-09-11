from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Literal

import anndata as an
import numpy as np
from mudata import MuData
from numpy.typing import NDArray
from pandas import DataFrame, Series, concat
from tqdm import tqdm

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._constants import DeMethod
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._types import get_obs
from liana.method.sc._liana_pipe import MdataKwargs, SpatialKwargs, liana_pipe

type ScoreValues = NDArray[np.floating] | Series
"""One column of scores, as a method's scoring function returns it."""

type ScoreFn = Callable[..., tuple[ScoreValues | None, ScoreValues | None]]
"""Turns ligand/receptor statistics into a (magnitude, specificity) pair.

Either element is `None` for a method that reports only one of the two.
"""


class MethodMeta:
    """
    A Class used to store Method Metadata

    Parameters
    ----------
    method_name
        Name of the Method
    complex_cols
        Columns relevant for protein complexes
    add_cols
        Additional columns required by the method
    fun
        Interaction Scoring function
    magnitude
        Name of the `magnitude` Score (None if not present)
    magnitude_ascending
        Whether to rank `magnitude` in ascending manner (None if not relevant)
    specificity
        Name of the `specificity` Score if Present (None if not present)
    specificity_ascending
        Whether to rank `magnitude` in ascending manner  (None if not relevant)
    permute
        Whether it requires permutations
    reference
        Publication reference in Harvard style

    Attributes
    ----------
    method_name
        Name of the Method
    complex_cols
        Columns relevant for protein complexes
    add_cols
        Additional columns required by the method
    fun
        Interaction Scoring function
    magnitude
        Name of the `magnitude` Score (None if not present)
    magnitude_ascending
        Whether to rank `magnitude` in ascending manner (None if not relevant)
    specificity
        Name of the `specificity` Score if Present (None if not present)
    specificity_ascending
        Whether to rank `magnitude` in ascending manner  (None if not relevant)
    permute
        Whether it requires permutations
    reference
        Publication reference in Harvard style
    """

    # `Method` and `AggregateClass` each define `__call__`; `by_sample` invokes it
    __call__: Callable[..., DataFrame | dict[str, DataFrame] | None]

    def __init__(
        self,
        method_name: str,
        complex_cols: list[str],
        add_cols: list[str],
        fun: ScoreFn | None,
        magnitude: str | None,
        magnitude_ascending: bool | None,
        specificity: str | None,
        specificity_ascending: bool | None,
        permute: bool,
        reference: str,
    ):
        self.method_name = method_name
        self.complex_cols = complex_cols
        self.add_cols = add_cols
        self.fun = fun
        self.magnitude = magnitude
        self.magnitude_ascending = magnitude_ascending
        self.specificity = specificity
        self.specificity_ascending = specificity_ascending
        self.permute = permute
        self.reference = reference

    def describe(self) -> None:
        """Briefly describes the method"""
        print(
            f"{self.method_name} uses `{self.magnitude}` and `{self.specificity}`"
            f" as measures of expression strength and interaction specificity, respectively"
        )

    def get_meta(self) -> DataFrame:
        """Returns method metadata as pandas row"""
        meta = DataFrame(
            [
                {
                    "Method Name": self.method_name,
                    "Magnitude Score": self.magnitude,
                    "Specificity Score": self.specificity,
                    "Reference": self.reference,
                }
            ]
        )
        return meta

    @d.dedent
    def by_sample(
        self,
        adata: an.AnnData | MuData,
        sample_key: str,
        key_added: str = K.uns_key,
        inplace: bool = V.inplace,
        verbose: bool | None | Literal["full"] = V.verbose,
        **kwargs: object,
    ) -> DataFrame | None:
        """
        Run a method by sample.

        Parameters
        ----------
        %(adata)s
        %(sample_key)s
        %(key_added)s
        %(inplace)s
        verbose
            Possible values: False, True, 'full', where 'full' will print the results for each sample,
            and True will only print the sample progress bar. Default is False.
        **kwargs
            Keyword arguments to pass to the method

        Raises
        ------
        ValueError
            If `sample_key` is not present in `adata.obs`

        Returns
        -------
        A pandas DataFrame with the results and a column sample is stored in `adata.uns[key_added]` if `inplace` is True,
        else the DataFrame is returned.
        """
        obs = get_obs(adata)
        if sample_key not in obs:
            raise ValueError(f"{sample_key} was not found in `adata.obs`.")

        full_verbose = verbose == "full"
        show_progress = bool(verbose)

        if not obs[sample_key].dtype.name == "category":
            _logg(
                f"Converting `{sample_key}` to categorical!",
                level="warn",
                verbose=None if verbose is None else show_progress,
            )
            obs[sample_key] = obs[sample_key].astype("category")

        samples = obs[sample_key].cat.categories

        by_sample: dict[Hashable, DataFrame] = {}

        progress_bar = tqdm(samples, disable=not show_progress)
        for sample in progress_bar:
            if show_progress:
                progress_bar.set_description(f"Now running: {sample}")

            subset = adata[obs[sample_key] == sample]
            if isinstance(subset, an.AnnData):
                temp: an.AnnData | MuData = subset.to_memory().copy() if subset.isbacked else subset.copy()
            elif isinstance(subset, MuData):
                # slicing a MuData yields a MuData, which the methods take just as well as an
                # AnnData; `MuData` has no `to_memory`, and its own `copy` explains what a backed
                # object needs instead
                temp = subset.copy()
            else:
                raise TypeError(f"Expected an AnnData or MuData slice, got {type(subset).__name__}.")

            sample_res = self(temp, inplace=False, verbose=full_verbose, **kwargs)
            if not isinstance(sample_res, DataFrame):  # only the consensus path returns a dict
                raise TypeError(f"Expected a DataFrame of results, got {type(sample_res).__name__}.")

            by_sample[sample] = sample_res

        liana_res = concat(by_sample).reset_index(level=1, drop=True).reset_index()
        liana_res = liana_res.rename({"index": sample_key}, axis=1)

        if inplace:
            adata.uns[key_added] = liana_res
        return None if inplace else liana_res


class Method(MethodMeta):
    """
    Ligand-Receptor Method Class

    Parameters
    ----------
    method
        Instance of metod metadata class

    Attributes
    ----------
    method
        Instance of metod metadata class
    """

    def __init__(self, _method: MethodMeta):
        super().__init__(
            method_name=_method.method_name,
            complex_cols=_method.complex_cols,
            add_cols=_method.add_cols,
            fun=_method.fun,
            magnitude=_method.magnitude,
            magnitude_ascending=_method.magnitude_ascending,
            specificity=_method.specificity,
            specificity_ascending=_method.specificity_ascending,
            permute=_method.permute,
            reference=_method.reference,
        )
        self._method = _method

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
        spatial_key: str | None = None,
        spatial_kwargs: SpatialKwargs | None = None,
        mdata_kwargs: MdataKwargs | None = None,
        inplace: bool = V.inplace,
        verbose: bool | None = V.verbose,
    ) -> DataFrame | None:
        """
        Run a ligand-receptor method.

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
        %(spatial_key)s
        %(spatial_kwargs)s
        %(mdata_kwargs)s
        %(inplace)s
        %(verbose)s

        Returns
        -------
        If ``inplace = False``, returns a `DataFrame` with ligand-receptor results
        Otherwise, modifies the ``adata`` object with the following key:

            - :attr:`anndata.AnnData.uns` ``[`key_added`]`` with the aforementioned DataFrame

        Examples
        --------
        Every method instance is called the same way; ``cellphonedb`` shown here:

        >>> import liana as li
        >>> adata = li.ds.generate_toy_adata()
        >>> li.mt.cellphonedb(adata, groupby="bulk_labels", n_perms=100)
        """
        if supp_columns is None:
            supp_columns = []
        if mdata_kwargs is None:
            mdata_kwargs = MdataKwargs()

        liana_res = liana_pipe(
            adata=adata,
            groupby=groupby,
            resource_name=resource_name,
            resource=resource,
            interactions=interactions,
            expr_prop=expr_prop,
            min_cells=min_cells,
            supp_columns=supp_columns,
            return_all_lrs=return_all_lrs,
            groupby_pairs=groupby_pairs,
            base=base,
            de_method=de_method,
            verbose=verbose,
            score=self._method,
            n_perms=n_perms,
            seed=seed,
            n_jobs=n_jobs,
            use_raw=use_raw,
            layer=layer,
            spatial_key=spatial_key,
            spatial_kwargs=spatial_kwargs,
            mdata_kwargs=mdata_kwargs,
        )
        if inplace:
            adata.uns[key_added] = liana_res
        return None if inplace else liana_res


def _show_methods(methods: Sequence[MethodMeta]) -> DataFrame:
    return concat([method.get_meta() for method in methods])
