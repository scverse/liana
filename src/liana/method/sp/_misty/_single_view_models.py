from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.tools.tools import add_constant


class _Predictor(Protocol):
    """A fitted estimator that can predict on a design matrix."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class SingleViewModel:
    """
    Base class for single view models. Subclasses should implement the fit method.

    Parameters
    ----------
    seed
        Pseudo-random number generator (PRNG) state seed.
    kwargs
        Other arguments used in a specific method. See the specific documentation in the corresponding child class.

    Attributes
    ----------
    seed : int
        The assigned initial state for the PRNG.
    kwargs : dict[str, Any]
        Keyword arguments passed to the fit function in a child class.
    model : object
        The fitted estimator, whose type depends on the subclass.
    predictions : numpy.ndarray
        Contains the resulting predictions in array form.
    importances : dict[str, float]
        Contains the importance scores of the different predictors.
    """

    def __init__(self, seed: int, **kwargs: Any) -> None:
        self.seed = seed
        self.kwargs = kwargs  # Store kwargs to be used in fit method
        # the estimator is model-specific; subclasses keep their concrete type locally
        self.model: object | None = None
        self.predictions: np.ndarray | None = None
        self.importances: dict[str, float] | None = None

    def fit(self, y: np.ndarray, X: np.ndarray, predictors: list[str], k_cv: int | None = None) -> None:
        """
        Fit the model to the data and store the predictions and importances.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Number of cross-validation folds. If None, no cross-validation is performed.

        Raises
        ------
        NotImplementedError
            Base class method, children classes replace it with their own method.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def _k_fold_predict(
        self, y: np.ndarray, X: np.ndarray, k_cv: int, fit_method: Callable[[np.ndarray, np.ndarray], _Predictor]
    ) -> np.ndarray:
        """
        Computes K-Fold cross-validation (CV)

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        k_cv
            Number of CV steps
        fit_method
            Model function to compute estimates

        Returns
        -------
        Matrix with the prediction results for each round of CV
        """
        predictions = np.zeros_like(y)
        kf = KFold(n_splits=k_cv, random_state=self.seed, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train = y[train_index]
            model = fit_method(y_train, X_train)
            y_pred = model.predict(X_test)
            predictions[test_index] = y_pred.flatten()
        return predictions


class RandomForestModel(SingleViewModel):
    """Random forest model (from sklearn) using out-of-bag predictions for feature importances. Inherits from `SingleViewModel`"""

    def fit(self, y: np.ndarray, X: np.ndarray, predictors: list[str], k_cv: int | None = None) -> None:
        """
        Fits a Random Forest (RF) model.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Not used
        """
        forest = RandomForestRegressor(oob_score=True, random_state=self.seed, **self.kwargs)
        forest.fit(X, y)
        self.model = forest
        self.predictions = forest.oob_prediction_
        self.importances = dict(zip(predictors, forest.feature_importances_, strict=False))


class LinearModel(SingleViewModel):
    """Linear regression model using `statsmodels.OLS` for feature importances, and `cross_val_predict` with `sklearn.LinearRegression` for predictions. Inherits from `SingleViewModel`"""

    def fit(self, y: np.ndarray, X: np.ndarray, predictors: list[str], k_cv: int | None = None) -> None:
        """
        Fits a Linear Model (LM) model.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Number of cross-validation folds. If None, no cross-validation is performed.
        """
        # NOTE: read, don't pop -- `fit` is called once per target on the same
        # instance, so popping would apply `n_jobs` to the first target only.
        # Folds are few and each fit is cheap, so serial is the sane default:
        # `-1` would fork a worker per core to cross-validate a linear model.
        n_jobs = self.kwargs.get("n_jobs", 1)
        ols_kwargs = {k: v for k, v in self.kwargs.items() if k != "n_jobs"}
        model = LinearRegression(n_jobs=1)
        self.predictions = cross_val_predict(
            model, X, y, cv=KFold(n_splits=k_cv, random_state=self.seed, shuffle=True), n_jobs=n_jobs
        )
        X = add_constant(X)
        model_full = OLS(y, X, **ols_kwargs).fit()
        self.importances = dict(zip(predictors, model_full.tvalues[1:], strict=False))

    def _fit_ols(self, y: np.ndarray, X: np.ndarray) -> _Predictor:
        fitted: _Predictor = LinearRegression(**self.kwargs).fit(y=y, X=X)
        return fitted


class RobustLinearModel(SingleViewModel):
    """Robust linear regression model using `statsmodels.RLM`. Inherits from `SingleViewModel`"""

    def fit(self, y: np.ndarray, X: np.ndarray, predictors: list[str], k_cv: int | None = None) -> None:
        """
        Fits a robust linear model.

        Parameters
        ----------
        y
            Target variable
        X
            Feature matrix
        predictors
            List of feature names
        k_cv
            Number of cross-validation folds. If None, no cross-validation is performed.
        """
        if k_cv is None:
            raise ValueError("`k_cv` must be provided for the robust linear model.")
        design = add_constant(X)
        self.predictions = self._k_fold_predict(y, design, k_cv, self._fit_robust)
        model_full = RLM(y, design, **self.kwargs).fit()
        self.importances = dict(zip(predictors, model_full.tvalues[1:], strict=False))

    def _fit_robust(self, y: np.ndarray, X: np.ndarray) -> _Predictor:
        fitted: _Predictor = RLM(y, X, **self.kwargs).fit()
        return fitted
