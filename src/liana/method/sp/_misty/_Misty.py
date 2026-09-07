from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from fast_array_utils.conv import to_dense
from fast_array_utils.types import CSBase
from mudata import MuData
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm

from liana._core._common import _logg
from liana._core._constants import DefaultValues as V
from liana._core._constants import Keys as K
from liana._core._docs import d
from liana._core._pipe_utils._pre import _choose_mtx_rep
from liana._core._types import MatrixLike, _to_matrix, get_obs, get_x
from liana.method.sp._misty._single_view_models import SingleViewModel


@d.dedent
class MistyData(MuData):
    """
    MistyData Class used to construct multi-view objects.

    Construct a MistyData object from a dictionary of views (anndatas).

    Parameters
    ----------
    data
        Dictionary of views (`AnnData`s) or a `MuData` object. Note that only the `data.X` attribute is used.
        An intra-view called "intra" is required.
    obs
        DataFrame of observations. If None, the obs of the intra-view is used.
    %(spatial_key)s
    enforce_obs
        If True, the number of observations in each extra-view must match the intra-view.
        Then the connectivities are stored in the .obsp attribute, while the weighted matrix is stored in .layers['weighted'].
        If False, the connectivities are stored in the .obsm attribute, while the weighted matrix is transposed and stored in .varm['weighted'].
    **kwargs
        Keyword arguments passed to the MuData Super class

    Attributes
    ----------
    view_names
        List of names of the different views
    spatial_key
        Key in `data.obsm` containing the spatial coordinates.
    enforce_obs
        See parameter with the same name.

    Examples
    --------
    Views are `AnnData`s that share observations. The `'intra'` view holds the
    targets to be predicted; every other view is a spatial context and must carry
    its own connectivities in `.obsp['spatial_connectivities']`:

    >>> import liana as li
    >>> adata = li.ds.generate_toy_spatial()
    >>> adata = adata[:, adata.var_names[:5]].copy()
    >>> extra = adata.copy()
    >>> extra.obsp["spatial_connectivities"] = li.pp.spatial_neighbors(
    ...     extra, bandwidth=200, set_diag=True, inplace=False
    ... )
    >>> misty = li.mt.MistyData({"intra": adata.copy(), "extra": extra})

    Each extra view's expression is multiplied by its connectivities on
    construction, so that a predictor is a *neighbourhood* value rather than the
    spot's own. :func:`liana.mt.genericMistyData` and
    :func:`liana.mt.lrMistyData` build the views for the two most common
    designs. Call the object to fit the model -- see
    :func:`liana.mt.MistyData.__call__`.
    """

    def __init__(
        self,
        data: dict[str, AnnData] | MuData,
        obs: pd.DataFrame | None = None,
        spatial_key: str = K.spatial_key,
        enforce_obs: bool = True,
        **kwargs: Any,
    ) -> None:
        source = data if isinstance(data, MuData) else None
        views: Mapping[str, AnnData]
        if source is not None:
            views = {name: view for name, view in source.mod.items() if isinstance(view, AnnData)}
        elif isinstance(data, MuData):  # unreachable: `source` covers the MuData case
            raise TypeError("`data` must be a mapping of views or a MuData.")
        else:
            views = data

        super().__init__(views, **kwargs)

        # preserve container-level attributes that MuData drops when rebuilt from .mod
        if source is not None:
            for attr in ("uns", "obsm", "varm", "obsp", "varp"):
                setattr(self, attr, getattr(source, attr))

        self.view_names = list(self.mod.keys())
        self.spatial_key = spatial_key
        self.enforce_obs = enforce_obs
        self._check_views()
        self.obs = obs if obs is not None else get_obs(self._view("intra"))

    def _view(self, view_name: str) -> AnnData:
        """Return one view, checked to be an AnnData rather than a nested MuData."""
        view = self.mod[view_name]
        if not isinstance(view, AnnData):
            raise TypeError(f"view '{view_name}' must be an AnnData, got {type(view).__name__}.")
        return view

    def _check_views(self) -> None:
        if "intra" not in self.view_names:
            raise ValueError("views must contain an intra view")

        for view in self.view_names:
            if view == "intra":
                continue
            current, intra = self._view(view), self._view("intra")
            if self.enforce_obs:
                if f"{self.spatial_key}_connectivities" not in current.obsp.keys():
                    raise ValueError(f"view {view} does not contain `{self.spatial_key}_connectivities` key in .obsp")
                if current.shape[0] != intra.shape[0]:
                    raise ValueError(
                        f"view {view} has {current.shape[0]} observations, "
                        + f"but the intra-view has {intra.shape[0]} observations"
                    )
            elif f"{self.spatial_key}_connectivities" not in current.obsm.keys():
                raise ValueError(f"view {view} does not contain `{self.spatial_key}_connectivities` key in .obsm")

            self._set_weighted_matrix(view)

    def _set_weighted_matrix(self, view_name: str) -> None:
        view = self._view(view_name)
        X = get_x(view)
        if self.enforce_obs:
            connectivities = view.obsp[f"{self.spatial_key}_connectivities"]
            view.layers["weighted"] = _to_matrix(connectivities @ X, what="weighted layer")
        else:
            weights = _to_matrix(view.obsm[f"{self.spatial_key}_connectivities"], what="spatial connectivities").T
            weighted = (weights @ X).T
            view.varm["weighted"] = weighted.tocsr() if isinstance(weighted, CSBase) else np.asarray(weighted)

    def get_weighted_matrix(self, view_name: str, predictors: list[str] | None = None) -> MatrixLike:
        """
        Returns the weighted matrix for a given set of predictors in a view.

        Parameters
        ----------
        view_name
            Name of the view of interest.
        predictors
            List of predictors from which to retrieve the weights.

        Returns
        -------
        Weighted matrix of the requested view and predictors. If no predictors are provided, returns the variable names.
        """
        view = self._view(view_name)
        selected = view.var_names if predictors is None else predictors
        subset = view[:, selected]

        if self.enforce_obs:
            return _to_matrix(subset.layers["weighted"], what="layers['weighted']")
        return _to_matrix(subset.varm["weighted"], what="varm['weighted']").T

    @d.dedent
    def __call__(
        self,
        model: type[SingleViewModel],
        bypass_intra: bool = False,
        predict_self: bool = False,
        maskby: str | None = None,
        k_cv: int = 10,
        alphas: float | NDArray[np.floating] | list[float] = np.array([0.1, 1, 10]),
        seed: int = V.seed,
        inplace: bool = V.inplace,
        verbose: bool = V.verbose,
        **kwargs: Any,
    ) -> None | tuple[pd.DataFrame, pd.DataFrame]:
        """
        A Multi-view Learning for dissecting Spatial Transcriptomics data (MISTy) model.

        Parameters
        ----------
        model
            Single-view model of class SingleViewModel. Default options are RandomForestModel, LinearModel, and RobustLinearModel
            available via ``liana.method.sp._misty._single_view_models``.
        bypass_intra
            Whether to bypass modeling the intraview via leave-one-feature-out (LOFO).
            In other words, whether to bypass modelling each target by LOFO within the same spots.
        predict_self
            Whether to predict self-interactions. These are determined purely by the feature names.
        maskby
            Column in the .obs attribute used to group or mask observations in the intra-view
            If None, all cells are considered as one group.
        k_cv
            Number of folds for cross-validation used in the multi-view model,
            and single-view models if model is 'linear'.
        alphas
            List of alpha values used to choose from, that control the strength of the ridge regression,
            used for the multi-view part of the model. Only used if there are more than 2 views being modeled (including intra).
        %(seed)s
        %(inplace)s
        %(verbose)s
        **kwargs
            Keyword arguments passed to the Regressors. Note that random_state is already set via ``seed``.
            ``n_jobs`` is instead used to cross-validate each target and defaults to 1, as the
            folds are few and each fit is cheap; raise it only for expensive regressors.

        Returns
        -------
        If inplace is True, two DataFrames are written to `misty.uns`. `'target_metrics'` is one row per
        target: how well the intra view alone explains it (`intra_R2`), how well all
        views together do (`multi_R2`), what the extra views add (`gain_R2`), and each
        view's contribution. `'interactions'` is one row per predictor-target pair per
        view, with the importance the model gave it.

        Otherwise the two DataFrames are returned, one for target metrics and one for importances.

        Examples
        --------
        Each variable of the `'intra'` view is modelled in turn, from the other
        intra-view variables and from every other view:

        >>> import liana as li
        >>> adata = li.ds.generate_toy_spatial()
        >>> adata = adata[:, adata.var_names[:5]].copy()
        >>> misty = li.mt.genericMistyData(intra=adata, bandwidth=200, set_diag=True)
        >>> misty(model=li.mt.sp.LinearModel)
        """
        fitted_model = model(seed, **kwargs)
        view_str = list(self.view_names)
        intra = self._view("intra")
        obs_masks = _create_obs_masks(intra, maskby)

        if bypass_intra:
            view_str.remove("intra")

        targets_list, importances_list = [], []
        intra_features = intra.var_names.to_list()
        progress_bar = tqdm(intra_features, disable=not verbose)

        for target in progress_bar:
            for intra_group in obs_masks.keys():
                msk = obs_masks[intra_group]
                importance_dict: dict[str, dict[str, float] | None] = {}
                if verbose:
                    d = f"Now learning: {target}" + (f" masked by {intra_group}" if intra_group is not None else "")
                    progress_bar.set_description(d)

                predictors_nonself, insert_index = _get_nonself(target, intra_features)
                y = _choose_mtx_rep(intra[msk, target]).toarray().ravel()
                X = _choose_mtx_rep(intra[msk, predictors_nonself]).toarray()

                if not bypass_intra:
                    fitted_model.fit(
                        y=y,
                        X=X,
                        predictors=predictors_nonself,
                        k_cv=k_cv,
                    )
                    predictions_intra, importance_dict["intra"] = fitted_model.predictions, fitted_model.importances

                    intra_importances = importance_dict["intra"]
                    if insert_index is not None and predict_self and intra_importances is not None:
                        # add self-interactions as nan
                        intra_importances[target] = np.nan

                # store the predictions for each view to construct predictor matrix for meta model
                predictions_list: list[np.ndarray] = []

                if not bypass_intra:
                    if predictions_intra is None:
                        raise RuntimeError("the intra model produced no predictions")
                    predictions_list.append(predictions_intra)

                # model the juxta and paraview (if applicable)
                for view_name in [v for v in view_str if v != "intra"]:
                    extra = self._view(view_name)

                    extra_features = extra.var_names.to_list()
                    _predictors, _ = (
                        _get_nonself(target, extra_features) if not predict_self else (extra_features, None)
                    )

                    X = to_dense(self.get_weighted_matrix(view_name, _predictors))
                    X = X[msk, :]
                    fitted_model.fit(
                        y=y,
                        X=X,
                        predictors=_predictors,
                        k_cv=k_cv,
                    )
                    predictions_extra, importance_dict[view_name] = fitted_model.predictions, fitted_model.importances

                    if predictions_extra is None:
                        raise RuntimeError(f"the {view_name} model produced no predictions")
                    predictions_list.append(predictions_extra)

                target_metrics = _multi_model(
                    y,
                    np.column_stack(predictions_list),
                    intra_group,
                    bypass_intra,
                    view_str,
                    target,
                    k_cv,
                    alphas,
                    seed,
                )
                targets_list.append(target_metrics)

                importances_df = _format_importances(
                    target=target, intra_group=intra_group, importance_dict=importance_dict
                )
                importances_list.append(importances_df)

        target_metrics, importances = _concat_dataframes(targets_list, importances_list, view_str)

        if inplace:
            self.uns[K.target_metrics] = target_metrics
            self.uns[K.interactions] = importances
            return None
        else:
            return target_metrics, importances


def _create_dict(**kwargs: object) -> dict[str, object]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _format_targets(
    target: str,
    intra_group: str | None,
    view_str: list[str],
    intra_r2: float,
    multi_r2: float,
    coefs: np.ndarray,
) -> pd.DataFrame:
    d = _create_dict(
        target=target,
        intra_group=intra_group,
        intra_R2=intra_r2,
        multi_R2=multi_r2,
        gain_R2=multi_r2 - intra_r2,
    )

    target_df = pd.DataFrame(d, index=[0])
    target_df[view_str] = coefs

    return target_df


def _format_importances(
    target: str,
    intra_group: str | None,
    importance_dict: dict[str, dict[str, float] | None],
) -> pd.DataFrame:

    importances_df = pd.DataFrame(importance_dict).reset_index().rename(columns={"index": "predictor"})
    importances_df[["target", "intra_group"]] = target, intra_group

    return importances_df


def _concat_dataframes(
    targets_list: list[pd.DataFrame],
    importances_list: list[pd.DataFrame],
    view_str: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_metrics = pd.concat(targets_list, axis=0, ignore_index=True)
    importances = pd.concat(importances_list, axis=0, ignore_index=True)
    importances = pd.melt(
        importances,
        id_vars=["target", "predictor", "intra_group"],
        value_vars=view_str,
        var_name="view",
        value_name="importances",
    )

    # drop intra and extra group columns if they are all None
    importances = importances.dropna(axis=1, how="all")
    importances = importances.dropna(axis=0)

    return target_metrics, importances


def _multi_model(
    y: np.ndarray,
    predictions: np.ndarray,
    intra_group: str | None,
    bypass_intra: bool,
    view_str: list[str],
    target: str,
    k_cv: int,
    alphas: float | NDArray[np.floating] | list[float],
    seed: int,
) -> pd.DataFrame:
    n_views = len(view_str)

    if (predictions.shape[0] < k_cv) or (y.var() == 0.0):
        if predictions.shape[0] < k_cv:
            warning_message = f"Number of samples is less than k_cv, {target} metrics set to NaN"
        else:
            warning_message = f"Variance of '{target}' is 0.0, metrics set to NaN"

        _logg(warning_message, verbose=True, level="warn")
        return _format_targets(target, intra_group, view_str, np.nan, np.nan, np.repeat(np.nan, n_views))

    kf = KFold(n_splits=k_cv, shuffle=True, random_state=seed)
    R2_vec_intra, R2_vec_multi = np.zeros(k_cv), np.zeros(k_cv)
    coef_mtx = np.zeros((k_cv, n_views))

    model = RidgeCV(alphas=alphas) if n_views > 2 else LinearRegression()

    for cv_idx, (train_index, test_index) in enumerate(kf.split(predictions)):
        multi_model = model.fit(X=predictions[train_index], y=y[train_index])
        R2_vec_multi[cv_idx] = multi_model.score(X=predictions[test_index], y=y[test_index])
        coef_mtx[cv_idx, :] = multi_model.coef_

        if not bypass_intra:
            pred_train = predictions[train_index, 0].reshape(-1, 1)
            pred_test = predictions[test_index, 0].reshape(-1, 1)

            intra_model = model.fit(X=pred_train, y=y[train_index])
            R2_vec_intra[cv_idx] = intra_model.score(X=pred_test, y=y[test_index])

    # format R2s
    intra_r2 = R2_vec_intra.mean().clip(min=0) if not bypass_intra else 0
    multi_r2 = R2_vec_multi.mean().clip(min=0)

    # format coefficients
    coefs = coef_mtx.mean(axis=0).clip(min=0)
    coefs = coefs / coefs.sum()

    # format metrics to a dataframe
    target_metrics = _format_targets(target, intra_group, view_str, float(intra_r2), float(multi_r2), coefs)

    return target_metrics


def _get_nonself(target: str, predictors: list[str]) -> tuple[list[str], int | None]:
    if target in predictors:
        insert_idx = np.where(np.array(predictors) == target)[0][0]
        predictors_subset = predictors.copy()
        predictors_subset.pop(insert_idx)
    else:
        predictors_subset = predictors
        insert_idx = None
    return predictors_subset, insert_idx


def _create_obs_masks(intra: AnnData, maskby: str | None) -> dict[str | None, np.ndarray]:
    obs_masks: dict[str | None, np.ndarray] = {}
    obs = get_obs(intra)
    # if maskby is a column of only boleans take it as is
    if maskby is None:
        obs_masks[None] = np.ones(intra.shape[0], dtype=bool)
    elif obs[maskby].dtype == bool:
        obs_masks[None] = obs[maskby].to_numpy()
    # else if maskby is column of strings convert to categorical
    elif obs[maskby].dtype == "category":
        for intra_group in obs[maskby].cat.categories:
            obs_masks[intra_group] = (obs[maskby] == intra_group).to_numpy()
    else:
        raise ValueError(f"maskby column {maskby} must be a column of booleans or categorical")

    return obs_masks
