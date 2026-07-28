from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import load_market_data
from .features import PreparedData, build_features
from .model import RidgeRanker


@dataclass(frozen=True)
class SelectionDiagnostics:
    score_date: str
    training_start: str
    training_end: str
    training_days: int
    training_rows: int
    eligible_stocks: int
    selected_stocks: int
    active_features: int
    industry_cap: int


@dataclass(frozen=True)
class SelectionResult:
    candidates: pd.DataFrame
    scored_universe: pd.DataFrame
    diagnostics: SelectionDiagnostics


@dataclass(frozen=True)
class BacktestResult:
    periods: pd.DataFrame
    holdings: pd.DataFrame
    summary: dict[str, float | int | str | None]


class CandidateSelector:
    def __init__(self, config: AppConfig):
        self.config = config

    def prepare(self, market_data: pd.DataFrame) -> PreparedData:
        return build_features(market_data, self.config)

    def prepare_from_csv(self, input_path: str | Path) -> PreparedData:
        market_data = load_market_data(input_path, self.config)
        return self.prepare(market_data)

    def _resolve_score_date(
        self, prepared: PreparedData, as_of: str | pd.Timestamp | None
    ) -> tuple[pd.Timestamp, pd.DatetimeIndex, int]:
        dates = pd.DatetimeIndex(
            sorted(pd.to_datetime(prepared.frame["date"].unique()))
        )
        if len(dates) == 0:
            raise ValueError("No dates are available after loading the input")

        if as_of is None:
            score_date = pd.Timestamp(dates[-1])
        else:
            requested = pd.Timestamp(as_of)
            available = dates[dates <= requested]
            if len(available) == 0:
                raise ValueError(f"No trading date is available on or before {requested.date()}")
            score_date = pd.Timestamp(available[-1])
        score_index = int(dates.get_loc(score_date))
        return score_date, dates, score_index

    def _fit_and_score(
        self,
        prepared: PreparedData,
        as_of: str | pd.Timestamp | None,
    ) -> tuple[pd.DataFrame, RidgeRanker, pd.Timestamp, pd.Timestamp, int]:
        score_date, dates, score_index = self._resolve_score_date(prepared, as_of)
        horizon = self.config.features.prediction_horizon
        training_end_index = score_index - horizon - 1
        if training_end_index < 0:
            raise ValueError("Not enough history to create leakage-safe training labels")

        lookback = self.config.model.train_lookback_days
        training_start_index = max(0, training_end_index - lookback + 1)
        training_dates = dates[training_start_index : training_end_index + 1]
        if len(training_dates) < self.config.model.min_train_days:
            raise ValueError(
                f"Need at least {self.config.model.min_train_days} training dates; "
                f"only {len(training_dates)} are available before {score_date.date()}"
            )

        frame = prepared.frame
        train_mask = (
            frame["date"].between(training_dates[0], training_dates[-1])
            & frame["eligible"]
            & frame["target_rank"].notna()
        )
        train = frame.loc[train_mask]
        if len(train) < self.config.model.min_train_rows:
            raise ValueError(
                f"Need at least {self.config.model.min_train_rows:,} training rows; "
                f"only {len(train):,} passed the filters"
            )

        score_mask = frame["date"].eq(score_date) & frame["eligible"]
        scored = frame.loc[score_mask].copy()
        if scored.empty:
            raise ValueError(f"No eligible stocks on {score_date.date()}")

        model = RidgeRanker(
            alpha=self.config.model.ridge_alpha,
            winsor_quantile=self.config.model.target_winsor_quantile,
        )
        model.fit(
            train[prepared.feature_columns].to_numpy(dtype=float),
            train["target_rank"].to_numpy(dtype=float),
        )
        scored["model_score"] = model.predict(
            scored[prepared.feature_columns].to_numpy(dtype=float)
        )
        scored["alpha_percentile"] = scored["model_score"].rank(
            method="average", pct=True
        )

        volatility_percentile = (
            scored["volatility_20_cs"].fillna(0.0).clip(-0.5, 0.5) + 0.5
        )
        illiquidity_percentile = (
            scored["amihud_20_cs"].fillna(0.0).clip(-0.5, 0.5) + 0.5
        )
        selection = self.config.selection
        scored["selection_score"] = (
            scored["alpha_percentile"]
            - selection.volatility_penalty * volatility_percentile
            - selection.illiquidity_penalty * illiquidity_percentile
        )
        scored = scored.sort_values(
            ["selection_score", "code"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        scored["global_rank"] = np.arange(1, len(scored) + 1)
        return (
            scored,
            model,
            pd.Timestamp(training_dates[0]),
            pd.Timestamp(training_dates[-1]),
            len(training_dates),
        )

    def _select_with_constraints(
        self,
        scored: pd.DataFrame,
        previous_codes: Iterable[str] | None,
    ) -> tuple[pd.DataFrame, int]:
        selection = self.config.selection
        target_count = min(selection.top_k, len(scored))
        industry_cap = max(
            1, math.ceil(selection.top_k * selection.max_industry_fraction)
        )
        previous = {str(code).strip().zfill(6) for code in (previous_codes or [])}
        buffer_rank = math.ceil(
            selection.top_k * selection.buffer_exit_multiplier
        )

        priority_rows: list[tuple[int, str]] = []
        for index, row in scored.iterrows():
            if row["code"] in previous and int(row["global_rank"]) <= buffer_rank:
                priority_rows.append((index, "buffer_keep"))
        priority_indices = {item[0] for item in priority_rows}
        for index in scored.index:
            if index not in priority_indices:
                priority_rows.append((int(index), "new_top_rank"))

        selected_indices: list[int] = []
        selected_reasons: dict[int, str] = {}
        selected_codes: set[str] = set()
        industry_counts: dict[str, int] = {}

        for index, reason in priority_rows:
            if len(selected_indices) >= target_count:
                break
            row = scored.loc[index]
            code = str(row["code"])
            industry = str(row["industry"])
            if code in selected_codes:
                continue
            if industry_counts.get(industry, 0) >= industry_cap:
                continue
            selected_indices.append(index)
            selected_codes.add(code)
            selected_reasons[index] = reason
            industry_counts[industry] = industry_counts.get(industry, 0) + 1

        if len(selected_indices) < target_count:
            for index, row in scored.iterrows():
                if len(selected_indices) >= target_count:
                    break
                code = str(row["code"])
                if code in selected_codes:
                    continue
                selected_indices.append(int(index))
                selected_codes.add(code)
                selected_reasons[int(index)] = "industry_cap_relaxed"

        candidates = scored.loc[selected_indices].copy()
        candidates["selection_reason"] = [
            selected_reasons[int(index)] for index in selected_indices
        ]
        candidates["candidate_position"] = np.arange(1, len(candidates) + 1)
        return candidates, industry_cap

    def select_prepared(
        self,
        prepared: PreparedData,
        as_of: str | pd.Timestamp | None = None,
        previous_codes: Iterable[str] | None = None,
    ) -> SelectionResult:
        (
            scored,
            model,
            training_start,
            training_end,
            training_days,
        ) = self._fit_and_score(prepared, as_of)
        candidates, industry_cap = self._select_with_constraints(
            scored, previous_codes
        )
        diagnostics = SelectionDiagnostics(
            score_date=pd.Timestamp(scored["date"].iloc[0]).date().isoformat(),
            training_start=training_start.date().isoformat(),
            training_end=training_end.date().isoformat(),
            training_days=training_days,
            training_rows=int(model.diagnostics_.rows),
            eligible_stocks=len(scored),
            selected_stocks=len(candidates),
            active_features=int(model.diagnostics_.active_features),
            industry_cap=industry_cap,
        )
        return SelectionResult(
            candidates=candidates.reset_index(drop=True),
            scored_universe=scored.reset_index(drop=True),
            diagnostics=diagnostics,
        )

    def select_from_csv(
        self,
        input_path: str | Path,
        as_of: str | pd.Timestamp | None = None,
        previous_candidates_path: str | Path | None = None,
    ) -> SelectionResult:
        previous_codes: list[str] | None = None
        if previous_candidates_path is not None:
            previous = pd.read_csv(
                previous_candidates_path, dtype={"code": "string"}
            )
            if "code" not in previous.columns:
                raise ValueError("Previous candidate file must contain a 'code' column")
            previous_codes = previous["code"].dropna().astype(str).tolist()
        prepared = self.prepare_from_csv(input_path)
        return self.select_prepared(prepared, as_of, previous_codes)

    def backtest_prepared(self, prepared: PreparedData) -> BacktestResult:
        frame = prepared.frame
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"].unique())))
        horizon = self.config.features.prediction_horizon
        start_index = self.config.model.min_train_days + horizon + 1
        stop_index = len(dates) - horizon - 1
        if stop_index <= start_index:
            raise ValueError("Not enough dates for walk-forward backtesting")

        step = self.config.backtest.rebalance_every_days
        score_dates = list(dates[start_index:stop_index:step])
        max_periods = self.config.backtest.max_periods
        if max_periods is not None:
            score_dates = score_dates[-max_periods:]
        if not score_dates:
            raise ValueError("No score dates were produced for the backtest")

        period_records: list[dict[str, float | int | str]] = []
        holding_frames: list[pd.DataFrame] = []
        previous_codes: list[str] | None = None

        for score_date in score_dates:
            result = self.select_prepared(
                prepared, as_of=score_date, previous_codes=previous_codes
            )
            candidates = result.candidates.copy()
            scored = result.scored_universe
            previous_set = set(previous_codes or [])
            current_set = set(candidates["code"].astype(str))
            turnover = (
                np.nan
                if previous_codes is None
                else 1.0
                - len(previous_set & current_set)
                / max(len(previous_set), len(current_set), 1)
            )
            ic = scored["model_score"].rank(method="average").corr(
                scored["target_excess_return"].rank(method="average")
            )
            period_records.append(
                {
                    "date": pd.Timestamp(score_date).date().isoformat(),
                    "candidate_count": len(candidates),
                    "eligible_count": len(scored),
                    "basket_forward_return": candidates["forward_return"].mean(),
                    "basket_excess_return": candidates[
                        "target_excess_return"
                    ].mean(),
                    "positive_excess_rate": candidates[
                        "target_excess_return"
                    ].gt(0).mean(),
                    "precision_top_20pct": candidates["target_rank"].ge(0.30).mean(),
                    "spearman_ic": ic,
                    "turnover": turnover,
                }
            )
            holding_frames.append(
                candidates[
                    [
                        "date",
                        "code",
                        "industry",
                        "candidate_position",
                        "global_rank",
                        "selection_score",
                        "forward_return",
                        "target_excess_return",
                        "selection_reason",
                    ]
                ]
            )
            previous_codes = candidates["code"].astype(str).tolist()

        periods = pd.DataFrame(period_records)
        holdings = pd.concat(holding_frames, ignore_index=True)
        period_returns = periods["basket_forward_return"].dropna()
        annual_periods = 252.0 / step
        sharpe = (
            float(period_returns.mean() / period_returns.std(ddof=1) * np.sqrt(annual_periods))
            if len(period_returns) > 1 and period_returns.std(ddof=1) > 0
            else None
        )
        summary: dict[str, float | int | str | None] = {
            "start_date": str(periods["date"].iloc[0]),
            "end_date": str(periods["date"].iloc[-1]),
            "periods": len(periods),
            "top_k": self.config.selection.top_k,
            "prediction_horizon_days": horizon,
            "rebalance_every_days": step,
            "mean_basket_forward_return": float(
                periods["basket_forward_return"].mean()
            ),
            "mean_basket_excess_return": float(
                periods["basket_excess_return"].mean()
            ),
            "mean_precision_top_20pct": float(
                periods["precision_top_20pct"].mean()
            ),
            "mean_spearman_ic": float(periods["spearman_ic"].mean()),
            "mean_turnover": float(periods["turnover"].mean()),
            "period_return_sharpe": sharpe,
        }
        return BacktestResult(periods=periods, holdings=holdings, summary=summary)


def write_selection_result(
    result: SelectionResult, output_dir: str | Path
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    candidate_path = directory / "candidates.csv"
    universe_path = directory / "scored_universe.csv"
    diagnostics_path = directory / "selection_diagnostics.json"

    candidate_columns = [
        "date",
        "code",
        "ts_code",
        "name",
        "industry",
        "market",
        "exchange",
        "candidate_position",
        "global_rank",
        "selection_score",
        "alpha_percentile",
        "model_score",
        "selection_reason",
        "close",
        "avg_amount_liquidity",
        "volatility_20",
        "momentum_5",
        "momentum_20",
        "momentum_60",
        "amount_ratio_20",
        "price_position_20",
        "is_limit_up",
        "is_limit_down",
    ]
    available_candidate_columns = [
        column for column in candidate_columns if column in result.candidates.columns
    ]
    result.candidates[available_candidate_columns].to_csv(
        candidate_path, index=False
    )
    universe_columns = [
        "date",
        "code",
        "ts_code",
        "name",
        "industry",
        "market",
        "exchange",
        "global_rank",
        "selection_score",
        "alpha_percentile",
        "model_score",
        "close",
        "avg_amount_liquidity",
        "eligible",
    ]
    available_universe_columns = [
        column
        for column in universe_columns
        if column in result.scored_universe.columns
    ]
    result.scored_universe[available_universe_columns].to_csv(
        universe_path, index=False
    )
    diagnostics_path.write_text(
        json.dumps(asdict(result.diagnostics), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "candidates": candidate_path,
        "scored_universe": universe_path,
        "diagnostics": diagnostics_path,
    }


def write_backtest_result(
    result: BacktestResult, output_dir: str | Path
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    periods_path = directory / "backtest_periods.csv"
    holdings_path = directory / "backtest_holdings.csv"
    summary_path = directory / "backtest_summary.json"
    result.periods.to_csv(periods_path, index=False)
    result.holdings.to_csv(holdings_path, index=False)
    summary_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "periods": periods_path,
        "holdings": holdings_path,
        "summary": summary_path,
    }
