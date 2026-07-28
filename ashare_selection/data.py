from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig


REQUIRED_MARKET_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

OPTIONAL_DEFAULTS: dict[str, object] = {
    "industry": "UNKNOWN",
    "adj_factor": 1.0,
    "market_cap": np.nan,
    "turnover_rate": np.nan,
    "listing_days": np.nan,
    "is_st": False,
    "is_suspended": False,
    "is_limit_up": False,
    "is_limit_down": False,
}

BOOL_COLUMNS = ["is_st", "is_suspended", "is_limit_up", "is_limit_down"]
NUMERIC_OPTIONAL_COLUMNS = [
    "adj_factor",
    "market_cap",
    "turnover_rate",
    "listing_days",
]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)

    normalized = series.astype("string").str.strip().str.lower()
    true_values = {"1", "true", "t", "yes", "y", "是"}
    false_values = {"0", "false", "f", "no", "n", "否", "", "<na>"}
    unknown = sorted(set(normalized.dropna().unique()) - true_values - false_values)
    if unknown:
        raise ValueError(f"Unrecognized boolean values: {unknown[:10]}")
    return normalized.isin(true_values)


def _normalize_code(series: pd.Series) -> pd.Series:
    codes = series.astype("string").str.strip()
    numeric_six_digit = codes.str.fullmatch(r"\d{1,6}", na=False)
    codes.loc[numeric_six_digit] = codes.loc[numeric_six_digit].str.zfill(6)
    return codes


def load_market_data(path: str | Path, config: AppConfig) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data not found: {data_path}")

    date_col = config.data.date_col
    code_col = config.data.code_col
    frame = pd.read_csv(data_path, dtype={code_col: "string"}, low_memory=False)

    required = [date_col, code_col, *REQUIRED_MARKET_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    rename_map = {date_col: "date", code_col: "code"}
    if config.data.industry_col in frame.columns:
        rename_map[config.data.industry_col] = "industry"
    frame = frame.rename(columns=rename_map)

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        examples = frame.loc[frame["date"].isna()].head(5).index.tolist()
        raise ValueError(f"Invalid date values at rows: {examples}")
    frame["code"] = _normalize_code(frame["code"])
    if frame["code"].isna().any() or frame["code"].eq("").any():
        raise ValueError("Stock code cannot be empty")

    for column in REQUIRED_MARKET_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column, default in OPTIONAL_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default

    for column in NUMERIC_OPTIONAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOL_COLUMNS:
        frame[column] = _coerce_bool(frame[column])

    frame["industry"] = (
        frame["industry"].astype("string").fillna("UNKNOWN").replace("", "UNKNOWN")
    )
    frame["adj_factor"] = frame["adj_factor"].replace(0, np.nan).fillna(1.0)

    duplicates = frame.duplicated(["date", "code"], keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, ["date", "code"]].head(10)
        raise ValueError(
            "Duplicate date/code rows detected:\n" + sample.to_string(index=False)
        )

    frame = frame.sort_values(["code", "date"], kind="stable").reset_index(drop=True)
    observed_listing_days = frame.groupby("code", sort=False).cumcount() + 1
    frame["listing_days"] = frame["listing_days"].fillna(observed_listing_days)

    positive_price = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    nonnegative_activity = frame[["volume", "amount"]].ge(0).all(axis=1)
    frame["has_valid_market_data"] = positive_price & nonnegative_activity

    for column in ["open", "high", "low", "close"]:
        frame[f"adj_{column}"] = frame[column] * frame["adj_factor"]

    return frame

