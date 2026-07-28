from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitDiagnostics:
    rows: int
    total_features: int
    active_features: int
    target_mean: float
    target_std: float


class RidgeRanker:
    """Small dependency-free baseline for cross-sectional ranking.

    The interface is intentionally narrow so it can later be replaced by a
    LightGBM LambdaRank model without changing the surrounding pipeline.
    """

    def __init__(self, alpha: float = 20.0, winsor_quantile: float = 0.01):
        self.alpha = float(alpha)
        self.winsor_quantile = float(winsor_quantile)
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None
        self.active_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.diagnostics_: FitDiagnostics | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RidgeRanker":
        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float)
        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if len(x) != len(y):
            raise ValueError("features and target have inconsistent row counts")

        valid_target = np.isfinite(y)
        x = x[valid_target]
        y = y[valid_target]
        if len(y) < 2:
            raise ValueError("Not enough finite target observations")

        medians = np.zeros(x.shape[1], dtype=float)
        for column_index in range(x.shape[1]):
            finite = np.isfinite(x[:, column_index])
            if finite.any():
                medians[column_index] = float(np.median(x[finite, column_index]))
        x = np.where(np.isfinite(x), x, medians)

        means = x.mean(axis=0)
        scales = x.std(axis=0)
        active = np.isfinite(scales) & (scales > 1e-10)
        if not active.any():
            raise ValueError("All model features are constant")

        x_active = (x[:, active] - means[active]) / scales[active]
        if 0 < self.winsor_quantile < 0.5:
            lower, upper = np.quantile(
                y, [self.winsor_quantile, 1.0 - self.winsor_quantile]
            )
            y = np.clip(y, lower, upper)

        target_mean = float(y.mean())
        centered_target = y - target_mean
        gram = x_active.T @ x_active
        penalty = np.eye(gram.shape[0], dtype=float) * self.alpha
        coef_active = np.linalg.solve(
            gram + penalty, x_active.T @ centered_target
        )

        coef = np.zeros(x.shape[1], dtype=float)
        coef[active] = coef_active
        self.medians_ = medians
        self.means_ = means
        self.scales_ = np.where(scales > 1e-10, scales, 1.0)
        self.active_ = active
        self.coef_ = coef
        self.intercept_ = target_mean
        self.diagnostics_ = FitDiagnostics(
            rows=len(y),
            total_features=x.shape[1],
            active_features=int(active.sum()),
            target_mean=target_mean,
            target_std=float(y.std()),
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if any(
            item is None
            for item in (
                self.medians_,
                self.means_,
                self.scales_,
                self.active_,
                self.coef_,
                self.intercept_,
            )
        ):
            raise RuntimeError("Model must be fitted before prediction")

        x = np.asarray(features, dtype=float)
        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if x.shape[1] != len(self.coef_):
            raise ValueError("Prediction feature count does not match fitted model")
        x = np.where(np.isfinite(x), x, self.medians_)
        standardized = (x - self.means_) / self.scales_
        return self.intercept_ + standardized @ self.coef_

