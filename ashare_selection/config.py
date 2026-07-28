from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class DataConfig:
    date_col: str = "date"
    code_col: str = "code"
    industry_col: str = "industry"


@dataclass
class UniverseConfig:
    min_listing_days: int = 120
    min_price: float = 2.0
    min_avg_amount: float = 20_000_000.0
    liquidity_window: int = 20
    exclude_st: bool = True
    exclude_suspended: bool = True
    exclude_limit_up_for_buy: bool = True


@dataclass
class FeatureConfig:
    prediction_horizon: int = 5
    momentum_windows: list[int] = field(default_factory=lambda: [5, 10, 20, 60])
    volatility_windows: list[int] = field(default_factory=lambda: [20, 60])
    min_feature_history: int = 60


@dataclass
class ModelConfig:
    ridge_alpha: float = 20.0
    train_lookback_days: int = 500
    min_train_days: int = 240
    min_train_rows: int = 5_000
    target_winsor_quantile: float = 0.01


@dataclass
class SelectionConfig:
    top_k: int = 200
    max_industry_fraction: float = 0.15
    volatility_penalty: float = 0.05
    illiquidity_penalty: float = 0.05
    buffer_exit_multiplier: float = 1.5


@dataclass
class BacktestConfig:
    rebalance_every_days: int = 5
    max_periods: int | None = 60


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _merge_dataclass(instance: T, values: dict[str, Any]) -> T:
    valid_fields = {item.name: item for item in fields(instance)}
    unknown = sorted(set(values) - set(valid_fields))
    if unknown:
        raise ValueError(
            f"Unknown configuration keys for {type(instance).__name__}: {unknown}"
        )

    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Configuration section '{name}' must be an object")
            _merge_dataclass(current, value)
        else:
            setattr(instance, name, value)
    return instance


def validate_config(config: AppConfig) -> None:
    if config.features.prediction_horizon < 1:
        raise ValueError("prediction_horizon must be at least 1")
    if config.model.min_train_days < 20:
        raise ValueError("min_train_days must be at least 20")
    if config.model.train_lookback_days < config.model.min_train_days:
        raise ValueError("train_lookback_days cannot be smaller than min_train_days")
    if config.model.min_train_rows < 100:
        raise ValueError("min_train_rows must be at least 100")
    if config.selection.top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 < config.selection.max_industry_fraction <= 1:
        raise ValueError("max_industry_fraction must be in (0, 1]")
    if config.selection.buffer_exit_multiplier < 1:
        raise ValueError("buffer_exit_multiplier must be at least 1")
    if config.backtest.rebalance_every_days < 1:
        raise ValueError("rebalance_every_days must be positive")


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    if path is not None:
        config_path = Path(path)
        values = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise TypeError("Top-level configuration must be a JSON object")
        _merge_dataclass(config, values)
    validate_config(config)
    return config


def write_default_config(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(AppConfig().to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path

