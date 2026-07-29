from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import AppConfig
from .generated_features import (
    DEFAULT_GENERATED_FEATURE_INPUTS,
    FeatureDefinition,
    apply_generated_features,
    load_feature_definitions,
)


@dataclass(frozen=True)
class PreparedData:
    frame: pd.DataFrame
    feature_columns: list[str]


def _rolling_transform(
    grouped: pd.core.groupby.SeriesGroupBy,
    window: int,
    operation: str,
) -> pd.Series:
    if operation == "mean":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    if operation == "std":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).std(ddof=0)
        )
    if operation == "min":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).min()
        )
    if operation == "max":
        return grouped.transform(
            lambda values: values.rolling(window, min_periods=window).max()
        )
    raise ValueError(f"Unsupported rolling operation: {operation}")


def _cross_sectional_rank(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    ranked_columns: list[str] = []
    by_date = frame.groupby("date", sort=False)
    for column in columns:
        clean = frame[column].replace([np.inf, -np.inf], np.nan)
        ranked_name = f"{column}_cs"
        frame[ranked_name] = (
            clean.groupby(frame["date"], sort=False).rank(
                method="average", pct=True, na_option="keep"
            )
            - 0.5
        ).astype(np.float32)
        ranked_columns.append(ranked_name)
    return ranked_columns


def build_features(
    market_data: pd.DataFrame,
    config: AppConfig,
    generated_definitions: list[FeatureDefinition] | None = None,
) -> PreparedData:
    frame = market_data.copy()
    grouped = frame.groupby("code", sort=False, group_keys=False)
    eps = 1e-12

    frame["return_1"] = grouped["adj_close"].pct_change(fill_method=None)
    previous_close = grouped["adj_close"].shift(1)
    frame["overnight_gap"] = frame["adj_open"] / previous_close - 1.0
    frame["intraday_return"] = frame["adj_close"] / frame["adj_open"] - 1.0
    frame["range_pct"] = (frame["adj_high"] - frame["adj_low"]) / (
        previous_close.abs() + eps
    )

    momentum_windows = sorted(
        set(config.features.momentum_windows) | {5, 10, 20, 60}
    )
    volatility_windows = sorted(
        set(config.features.volatility_windows) | {20, 60}
    )
    raw_features = ["return_1", "overnight_gap", "intraday_return", "range_pct"]

    for window in momentum_windows:
        column = f"momentum_{window}"
        frame[column] = frame["adj_close"] / grouped["adj_close"].shift(window) - 1.0
        raw_features.append(column)

    return_grouped = frame.groupby("code", sort=False)["return_1"]
    for window in volatility_windows:
        column = f"volatility_{window}"
        frame[column] = _rolling_transform(return_grouped, window, "std")
        raw_features.append(column)

    amount_grouped = frame.groupby("code", sort=False)["amount"]
    volume_grouped = frame.groupby("code", sort=False)["volume"]
    close_grouped = frame.groupby("code", sort=False)["adj_close"]

    liquidity_window = config.universe.liquidity_window
    frame["avg_amount_liquidity"] = _rolling_transform(
        amount_grouped, liquidity_window, "mean"
    )
    frame["amount_ratio_20"] = frame["amount"] / (
        _rolling_transform(amount_grouped, 20, "mean") + eps
    )
    frame["volume_ratio_20"] = frame["volume"] / (
        _rolling_transform(volume_grouped, 20, "mean") + eps
    )

    rolling_low = _rolling_transform(close_grouped, 20, "min")
    rolling_high = _rolling_transform(close_grouped, 20, "max")
    frame["price_position_20"] = (frame["adj_close"] - rolling_low) / (
        rolling_high - rolling_low + eps
    )

    frame["daily_illiquidity"] = frame["return_1"].abs() / (
        frame["amount"].clip(lower=1.0)
    )
    illiquidity_grouped = frame.groupby("code", sort=False)["daily_illiquidity"]
    frame["amihud_20"] = (
        _rolling_transform(illiquidity_grouped, 20, "mean") * 100_000_000
    )

    frame["gap_volnorm_20"] = frame["overnight_gap"] / (
        frame["volatility_20"] + eps
    )
    frame["momentum_5_volnorm_20"] = frame["momentum_5"] / (
        frame["volatility_20"] * np.sqrt(5.0) + eps
    )
    frame["trend_5_20"] = (
        frame["momentum_5"] - frame["momentum_20"] * (5.0 / 20.0)
    ) / (frame["volatility_20"] * np.sqrt(5.0) + eps)
    frame["volatility_regime"] = frame["volatility_20"] / (
        frame["volatility_60"] + eps
    )
    frame["log_market_cap"] = np.log1p(frame["market_cap"].clip(lower=0))

    raw_features.extend(
        [
            "amount_ratio_20",
            "volume_ratio_20",
            "price_position_20",
            "amihud_20",
            "gap_volnorm_20",
            "momentum_5_volnorm_20",
            "trend_5_20",
            "volatility_regime",
            "log_market_cap",
            "turnover_rate",
        ]
    )
    raw_features = list(dict.fromkeys(raw_features))
    baseline_model_features = _cross_sectional_rank(frame, raw_features)

    generated_inputs = (
        set(DEFAULT_GENERATED_FEATURE_INPUTS) | set(raw_features)
    ) & set(frame.columns)
    if generated_definitions is None and config.features.generated_feature_path:
        generated_definitions = load_feature_definitions(
            config.features.generated_feature_path,
            allowed_names=generated_inputs,
        )
    generated_model_features: list[str] = []
    if generated_definitions:
        generated_columns = apply_generated_features(
            frame,
            generated_definitions,
            allowed_names=generated_inputs,
        )
        generated_model_features = _cross_sectional_rank(
            frame,
            generated_columns,
        )
        frame.drop(columns=generated_columns, inplace=True)
    model_features = [*baseline_model_features, *generated_model_features]

    feature_coverage = frame[baseline_model_features].notna().mean(axis=1)
    frame["feature_ready"] = (
        frame["listing_days"].ge(config.features.min_feature_history)
        & feature_coverage.ge(0.60)
    )

    universe = config.universe
    eligible = (
        frame["has_valid_market_data"]
        & frame["feature_ready"]
        & frame["listing_days"].ge(universe.min_listing_days)
        & frame["close"].ge(universe.min_price)
        & frame["avg_amount_liquidity"].ge(universe.min_avg_amount)
    )
    if universe.exclude_st:
        eligible &= ~frame["is_st"]
    if universe.exclude_suspended:
        eligible &= ~frame["is_suspended"]
    if universe.exclude_limit_up_for_buy:
        eligible &= ~frame["is_limit_up"]
    frame["eligible"] = eligible.fillna(False)

    horizon = config.features.prediction_horizon
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"].unique())))
    entry_date_map = {
        calendar[index]: calendar[index + 1]
        for index in range(max(0, len(calendar) - 1))
    }
    exit_date_map = {
        calendar[index]: calendar[index + horizon + 1]
        for index in range(max(0, len(calendar) - horizon - 1))
    }
    price_lookup = frame.set_index(["code", "date"])["adj_open"]
    entry_dates = frame["date"].map(entry_date_map)
    exit_dates = frame["date"].map(exit_date_map)
    entry_index = pd.MultiIndex.from_arrays(
        [frame["code"], entry_dates], names=["code", "date"]
    )
    exit_index = pd.MultiIndex.from_arrays(
        [frame["code"], exit_dates], names=["code", "date"]
    )
    entry_open = price_lookup.reindex(entry_index).to_numpy(dtype=float)
    exit_open = price_lookup.reindex(exit_index).to_numpy(dtype=float)
    frame["forward_return"] = exit_open / entry_open - 1.0
    frame["target_excess_return"] = frame["forward_return"] - frame.groupby(
        "date", sort=False
    )["forward_return"].transform("mean")
    frame["target_rank"] = (
        frame.groupby("date", sort=False)["target_excess_return"].rank(
            method="average", pct=True, na_option="keep"
        )
        - 0.5
    )

    frame = frame.sort_values(["date", "code"], kind="stable").reset_index(drop=True)
    return PreparedData(frame=frame, feature_columns=model_features)
