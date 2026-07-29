from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import lightgbm as lgb
except ImportError as error:  # pragma: no cover - exercised only in a broken environment.
    raise ImportError(
        "LightGBM is required. Install project dependencies with "
        "'python3 -m pip install -r requirements.txt'."
    ) from error


@dataclass(frozen=True)
class FitDiagnostics:
    rows: int
    total_features: int
    active_features: int
    target_mean: float
    target_std: float
    model_type: str
    trained_trees: int


class LightGBMReturnModel:
    """LightGBM regression model for cross-sectional future-return ranks."""

    MODEL_TYPE = "lightgbm_regression_l2"

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 500,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
        target_winsor_quantile: float = 0.01,
    ):
        self.target_winsor_quantile = float(target_winsor_quantile)
        self.params = {
            "objective": "regression_l2",
            "boosting_type": "gbdt",
            "metric": "l2",
            "learning_rate": float(learning_rate),
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "min_data_in_leaf": int(min_child_samples),
            "bagging_fraction": float(subsample),
            "bagging_freq": 1 if float(subsample) < 1.0 else 0,
            "feature_fraction": float(colsample_bytree),
            "lambda_l1": float(reg_alpha),
            "lambda_l2": float(reg_lambda),
            "seed": int(random_state),
            "bagging_seed": int(random_state),
            "feature_fraction_seed": int(random_state),
            "data_random_seed": int(random_state),
            "num_threads": 0 if int(n_jobs) == -1 else int(n_jobs),
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        self.n_estimators = int(n_estimators)
        self.active_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        self.model_: lgb.Booster | None = None
        self.diagnostics_: FitDiagnostics | None = None

    @staticmethod
    def _active_feature_mask(features: np.ndarray) -> np.ndarray:
        active = np.zeros(features.shape[1], dtype=bool)
        for column_index in range(features.shape[1]):
            values = features[:, column_index]
            finite_values = values[np.isfinite(values)]
            active[column_index] = (
                len(finite_values) >= 2
                and float(np.ptp(finite_values)) > 1e-10
            )
        return active

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "LightGBMReturnModel":
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(target, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if len(x) != len(y):
            raise ValueError("features and target have inconsistent row counts")
        if feature_names is not None and len(feature_names) != x.shape[1]:
            raise ValueError("feature_names count does not match feature columns")

        valid_target = np.isfinite(y)
        if not valid_target.all():
            x = x[valid_target]
            y = y[valid_target]
        if len(y) < 2:
            raise ValueError("Not enough finite target observations")

        invalid_features = ~np.isfinite(x)
        if invalid_features.any():
            x[invalid_features] = np.nan
        active = self._active_feature_mask(x)
        if not active.any():
            raise ValueError("All model features are constant or missing")

        if 0 < self.target_winsor_quantile < 0.5:
            lower, upper = np.quantile(
                y,
                [
                    self.target_winsor_quantile,
                    1.0 - self.target_winsor_quantile,
                ],
            )
            y = np.clip(y, lower, upper)

        if feature_names is None:
            all_feature_names = [f"feature_{index}" for index in range(x.shape[1])]
        else:
            all_feature_names = [str(name) for name in feature_names]
        active_feature_names = [
            name for name, is_active in zip(all_feature_names, active) if is_active
        ]

        train_set = lgb.Dataset(
            x[:, active],
            label=y,
            feature_name=active_feature_names,
            free_raw_data=True,
        )
        model = lgb.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=self.n_estimators,
        )

        self.active_ = active
        self.feature_names_ = all_feature_names
        self.model_ = model
        self.diagnostics_ = FitDiagnostics(
            rows=len(y),
            total_features=x.shape[1],
            active_features=int(active.sum()),
            target_mean=float(y.mean()),
            target_std=float(y.std()),
            model_type=self.MODEL_TYPE,
            trained_trees=int(model.current_iteration()),
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.active_ is None or self.model_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if x.shape[1] != len(self.active_):
            raise ValueError("Prediction feature count does not match fitted model")
        invalid_features = ~np.isfinite(x)
        if invalid_features.any():
            x[invalid_features] = np.nan
        return np.asarray(self.model_.predict(x[:, self.active_]), dtype=float)
