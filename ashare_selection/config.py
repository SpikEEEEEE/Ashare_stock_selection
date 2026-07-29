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
    generated_feature_path: str | None = None


@dataclass
class ModelConfig:
    train_lookback_days: int = 500
    min_train_days: int = 240
    min_train_rows: int = 5_000
    target_winsor_quantile: float = 0.01
    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = -1


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
class TushareConfig:
    token_env: str = "TUSHARE_TOKEN"
    cache_dir: str = "data/tushare_cache"
    request_interval_seconds: float = 0.13
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0
    refresh_last_trading_days: int = 2
    include_name_history: bool = True


@dataclass
class DeepSeekConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    thinking_enabled: bool = True
    temperature: float = 0.2
    max_tokens: int = 8_000
    timeout_seconds: float = 120.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    proposal_count: int = 30


@dataclass
class FeatureScreeningConfig:
    min_feature_coverage: float = 0.80
    min_screening_days: int = 120
    min_cross_sectional_stocks: int = 100
    min_abs_mean_ic: float = 0.002
    min_abs_ic_tstat: float = 1.0
    max_pairwise_correlation: float = 0.85
    correlation_sample_rows: int = 200_000
    max_selected_features: int = 30


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    tushare: TushareConfig = field(default_factory=TushareConfig)
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    feature_screening: FeatureScreeningConfig = field(
        default_factory=FeatureScreeningConfig
    )

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
    if not 0 <= config.model.target_winsor_quantile < 0.5:
        raise ValueError("target_winsor_quantile must be in [0, 0.5)")
    if config.model.n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    if config.model.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.model.num_leaves < 2:
        raise ValueError("num_leaves must be at least 2")
    if config.model.max_depth == 0 or config.model.max_depth < -1:
        raise ValueError("max_depth must be -1 or a positive integer")
    if config.model.min_child_samples < 1:
        raise ValueError("min_child_samples must be positive")
    if not 0 < config.model.subsample <= 1:
        raise ValueError("subsample must be in (0, 1]")
    if not 0 < config.model.colsample_bytree <= 1:
        raise ValueError("colsample_bytree must be in (0, 1]")
    if config.model.reg_alpha < 0:
        raise ValueError("reg_alpha cannot be negative")
    if config.model.reg_lambda < 0:
        raise ValueError("reg_lambda cannot be negative")
    if config.model.n_jobs != -1 and config.model.n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer")
    if config.selection.top_k < 1:
        raise ValueError("top_k must be positive")
    if not 0 < config.selection.max_industry_fraction <= 1:
        raise ValueError("max_industry_fraction must be in (0, 1]")
    if config.selection.buffer_exit_multiplier < 1:
        raise ValueError("buffer_exit_multiplier must be at least 1")
    if config.backtest.rebalance_every_days < 1:
        raise ValueError("rebalance_every_days must be positive")
    if config.tushare.request_interval_seconds < 0:
        raise ValueError("request_interval_seconds cannot be negative")
    if config.tushare.max_retries < 1:
        raise ValueError("max_retries must be positive")
    if config.tushare.retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")
    if config.tushare.refresh_last_trading_days < 0:
        raise ValueError("refresh_last_trading_days cannot be negative")
    if not config.deepseek.api_key_env.strip():
        raise ValueError("deepseek.api_key_env cannot be empty")
    if not config.deepseek.base_url.startswith("https://"):
        raise ValueError("deepseek.base_url must be an HTTPS URL")
    if not config.deepseek.model.strip():
        raise ValueError("deepseek.model cannot be empty")
    if not 0 <= config.deepseek.temperature <= 2:
        raise ValueError("deepseek.temperature must be in [0, 2]")
    if config.deepseek.max_tokens < 256:
        raise ValueError("deepseek.max_tokens must be at least 256")
    if config.deepseek.timeout_seconds <= 0:
        raise ValueError("deepseek.timeout_seconds must be positive")
    if config.deepseek.max_retries < 1:
        raise ValueError("deepseek.max_retries must be positive")
    if config.deepseek.retry_backoff_seconds < 0:
        raise ValueError("deepseek.retry_backoff_seconds cannot be negative")
    if config.deepseek.proposal_count < 1:
        raise ValueError("deepseek.proposal_count must be positive")
    if config.deepseek.proposal_count > 100:
        raise ValueError("deepseek.proposal_count cannot exceed 100")
    screening = config.feature_screening
    if not 0 < screening.min_feature_coverage <= 1:
        raise ValueError("min_feature_coverage must be in (0, 1]")
    if screening.min_screening_days < 2:
        raise ValueError("min_screening_days must be at least 2")
    if screening.min_cross_sectional_stocks < 2:
        raise ValueError("min_cross_sectional_stocks must be at least 2")
    if screening.min_abs_mean_ic < 0:
        raise ValueError("min_abs_mean_ic cannot be negative")
    if screening.min_abs_ic_tstat < 0:
        raise ValueError("min_abs_ic_tstat cannot be negative")
    if not 0 < screening.max_pairwise_correlation <= 1:
        raise ValueError("max_pairwise_correlation must be in (0, 1]")
    if screening.correlation_sample_rows < 100:
        raise ValueError("correlation_sample_rows must be at least 100")
    if screening.max_selected_features < 1:
        raise ValueError("max_selected_features must be positive")


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
