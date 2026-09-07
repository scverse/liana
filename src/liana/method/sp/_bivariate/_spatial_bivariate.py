from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from mudata import MuData
from scipy.sparse import csr_matrix

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._pipe_utils import assert_covered
from liana._core._pipe_utils._common import _get_props
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._types import copy_aligned, get_obs
from liana.method.sp._bivariate._global_functions import GlobalFunction, Weight
from liana.method.sp._bivariate._local_functions import LocalFunction
from liana.method.sp._utils import (
    _add_complexes_to_var,
    _check_instance,
    _handle_connectivity,
    _process_anndata,
    _process_mudata,
    _rename_means,
    _zscore,
)
from liana.resource.select_resource import _handle_resource


class SpatialBivariate:
    """
    A class for bivariate local spatial metrics.

    Parameters
    ----------
    %(x_name)s
    %(y_name)s

    Attributes
    ----------
    %(x_name)s
    %(y_name)s
    """

    @d.dedent
    def __call__(
        self,
        mdata: MuData | AnnData,
        local_name: str | None = "cosine",
        global_name: str | list[str] | None = None,
        resource_name: str | None = None,
        resource: pd.DataFrame | None = V.resource,
        interactions: list[tuple[str, str]] | None = V.interactions,
        connectivity_key: str = K.connectivity_key,
        mask_negatives: bool = False,
        add_categories: bool = False,
        n_perms: int | None = None,
        seed: int = V.seed,
        nz_prop: float = V.nz_prop,
        remove_self_interactions: bool = True,
        complex_sep: None | str = "_",
        xy_sep: str = V.lr_sep,
        verbose: bool = V.verbose,
        **kwargs: Any,
    ) -> AnnData | pd.DataFrame | None:
        """
        A method for bivariate local spatial metrics.

        Parameters
        ----------
        %(mdata)s
        %(local_name)s
        %(global_name)s
        %(resource_name)s
        %(resource)s
        %(interactions)s
        %(connectivity_key)s
        %(mask_negatives)s
        %(add_categories)s
        %(n_perms)s
        %(seed)s
        %(nz_prop)s
            Both features must pass the threshold to be considered in the analysis.
        remove_self_interactions
            Whether to remove self-interactions. `True` by default.
        complex_sep
            Separator to use for complex names.
        xy_sep
            Separator to use for interaction names.
        %(verbose)s
        **kwargs
            Additional keyword arguments.

            For an `AnnData` input:

            x_name
                Name of the x-variable. If passing a `resource` dataframe, this should
                match the first column. By default: 'ligand'.
            y_name
                Name of the y-variable. If passing a `resource` dataframe, this should
                match the second column. By default: 'receptor'.

            For a `MuData` input:

            x_mod
                Name of the modality to use for the x-axis.
            y_mod
                Name of the modality to use for the y-axis.
            x_name
                Name of the x-variable. If passing a `resource` dataframe, this should
                match the first column. By default: 'x'.
            y_name
                Name of the y-variable. If passing a `resource` dataframe, this should
                match the second column. By default: 'y'.
            x_use_raw: bool
                Whether to use the raw counts for the x-mod.
            y_use_raw: bool
                Whether to use the raw counts for y-mod.
            x_layer: str
                Layer to use for x-mod.
            y_layer: str
                Layer to use for y-mod.
            x_transform: bool
                Function to transform the x-mod.
            y_transform: bool
                Function to transform the y-mod.

        Raises
        ------
        ValueError
            If `n_perms` is not None or negative or if `mdata` is not a valid type.

        Returns
        -------
        An AnnData object, (optionally) with multiple layers which correspond
        categories/p-values, and the actual scores are stored in `.X`.
        Moreover, global stats are stored in ``.var``.

        Examples
        --------
        Relates each ligand to its receptor at every spot, given the spatial
        connectivities of :func:`liana.pp.spatial_neighbors`:

        >>> import liana as li
        >>> adata = li.ds.generate_toy_spatial()
        >>> lrdata = li.mt.bivariate(
        ...     adata, resource_name="consensus", local_name="morans", global_name="morans", n_perms=0
        ... )

        One column per ligand-receptor pair that passed the expression filters, named
        `'ligand^receptor'`. `n_perms=0` uses the analytical p-values available for
        Moran's R -- a positive integer runs that many permutations instead, `None`
        skips them.

        ``li.mt.bivariate.show_functions()`` lists the available `local_name` choices.
        Pass a `MuData` with `x_mod`/`y_mod` instead of an `AnnData` to relate two
        modalities.
        """
        if n_perms is not None and n_perms < 0:
            raise ValueError("n_perms must be None, 0 for analytical or > 0 for permutation")
        global_names = [global_name] if isinstance(global_name, str) else global_name
        # reject only when none of the requested statistics supports analytical p-values
        if n_perms == 0 and (
            local_name not in ["morans", None]
            or (global_names is not None and not any(name == "morans" for name in global_names))
        ):
            raise ValueError("An analytical solution is currently available only for Moran's R")

        local_fn = None if local_name is None else LocalFunction._get_instance(name=local_name)

        _check_instance(mdata)  # raises for anything other than AnnData/MuData
        if isinstance(mdata, MuData):
            adata, x_name, y_name = _process_mudata(mdata, complex_sep, verbose, **kwargs)
        else:
            adata, x_name, y_name = _process_anndata(mdata, complex_sep, verbose, **kwargs)

        resource = _handle_resource(
            interactions=interactions,
            resource=resource,
            resource_name=resource_name,
            x_name=x_name,
            y_name=y_name,
            verbose=verbose,
        )
        weight = _handle_connectivity(adata=adata, connectivity_key=connectivity_key)

        if complex_sep is not None:
            adata = _add_complexes_to_var(
                adata, np.union1d(resource[x_name].astype(str), resource[y_name].astype(str)), complex_sep=complex_sep
            )

        # filter_resource
        resource = resource[(np.isin(resource[x_name], adata.var_names)) & (np.isin(resource[y_name], adata.var_names))]

        self_interactions = resource[x_name] == resource[y_name]
        if self_interactions.any() & remove_self_interactions:
            _logg(f"Removing {self_interactions.sum()} self-interactions", verbose=verbose)
            resource = resource[~self_interactions]

        # get entities
        entities = np.union1d(np.unique(resource[x_name]), np.unique(resource[y_name]))
        assert_covered(entities, adata.var_names, verbose=verbose)

        # Filter to only include the relevant features
        adata = adata[:, np.intersect1d(entities, adata.var.index)]

        adata_x = _choose_mtx_rep(adata)
        xy_stats = (
            pd.DataFrame(
                {"means": np.asarray(adata_x.mean(axis=0)).ravel(), "props": _get_props(adata_x)}, index=adata.var_names
            )
            .reset_index()
            .rename(columns={"index": "gene"})
        )
        # join global stats to LRs from resource
        xy_stats = resource.merge(_rename_means(xy_stats, entity=x_name)).merge(_rename_means(xy_stats, entity=y_name))

        # filter according to props
        xy_stats = xy_stats[(xy_stats[f"{x_name}_props"] >= nz_prop) & (xy_stats[f"{y_name}_props"] >= nz_prop)]
        if xy_stats.empty:
            raise ValueError("No features with non-zero proportions")

        # create interaction column
        xy_stats["interaction"] = xy_stats[x_name] + xy_sep + xy_stats[y_name]

        x_mat = _choose_mtx_rep(adata[:, xy_stats[x_name]])
        y_mat = _choose_mtx_rep(adata[:, xy_stats[y_name]])

        if global_names is not None:
            for gname in global_names:
                global_fn = GlobalFunction.instances[gname]
                global_fn(
                    xy_stats,
                    x_mat=x_mat,
                    y_mat=y_mat,
                    weight=weight,
                    seed=seed,
                    n_perms=n_perms,
                    mask_negatives=mask_negatives,
                    verbose=verbose,
                )

        if local_name is None:
            return xy_stats

        # Calculate local scores
        if add_categories or mask_negatives:
            local_cats = self._categorize(
                x_mat=x_mat,
                y_mat=y_mat,
                weight=weight,
            )
        else:
            local_cats = None

        if local_fn is None:
            raise ValueError("`local_name` must be provided to compute local scores.")

        # get local scores
        local_scores, local_pvals = local_fn(
            x_mat=x_mat,
            y_mat=y_mat,
            weight=weight,
            seed=seed,
            n_perms=n_perms,
            mask_negatives=mask_negatives,
            verbose=verbose,
        )

        xy_stats.loc[:, ["mean", "std"]] = np.vstack([np.mean(local_scores, axis=0), np.std(local_scores, axis=0)]).T

        if mask_negatives:
            local_scores = np.where(local_cats != 1, 0, local_scores)
            if local_pvals is not None:
                local_pvals = np.where(local_cats != 1, 1, local_pvals)

        scores = AnnData(
            csr_matrix(local_scores),
            obs=get_obs(adata),
            var=xy_stats.set_index("interaction"),
            uns=dict(adata.uns),
        )
        copy_aligned(scores, obsm=adata.obsm, obsp=adata.obsp)

        if add_categories and local_cats is not None:
            scores.layers["cats"] = csr_matrix(local_cats)
        if local_pvals is not None:
            scores.layers["pvals"] = csr_matrix(local_pvals)

        return scores

    def _encode_cats(self, a: np.ndarray, weight: Weight) -> np.ndarray:
        if np.all(a >= 0):
            a = _zscore(a)
        weighted = weight @ a
        return np.where(weighted > 0, 1, np.where(weighted < 0, -1, np.nan))

    def _categorize(
        self,
        x_mat: csr_matrix,
        y_mat: csr_matrix,
        weight: Weight,
    ) -> np.ndarray:
        x_cats = self._encode_cats(x_mat.toarray(), weight)
        y_cats = self._encode_cats(y_mat.toarray(), weight)
        cats = x_cats + y_cats
        cats = np.where(cats == 2, 1, np.where(cats == 0, -1, 0))

        return cats

    def show_functions(self) -> pd.DataFrame:
        """
        Print information about all bivariate local metrics.

        Returns
        -------
        Table of the bivariate methods and their description.
        """
        funs = {
            function.name: {
                "metadata": function.metadata,
                "reference": function.reference,
            }
            for function in LocalFunction.instances.values()
        }
        return pd.DataFrame(funs).T.reset_index().rename(columns={"index": "name"})


bivariate = SpatialBivariate()
