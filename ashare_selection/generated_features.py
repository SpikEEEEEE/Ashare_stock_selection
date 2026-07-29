from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import AppConfig


FEATURE_FILE_SCHEMA_VERSION = 1
FEATURE_NAME_PATTERN = re.compile(r"^ai_[a-z][a-z0-9_]{2,59}$")
MAX_EXPRESSION_LENGTH = 512
MAX_AST_NODES = 128
MAX_AST_DEPTH = 12
MAX_LOOKBACK = 252

DEFAULT_GENERATED_FEATURE_INPUTS = {
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
    "amount",
    "market_cap",
    "turnover_rate",
    "return_1",
    "overnight_gap",
    "intraday_return",
    "range_pct",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "momentum_60",
    "volatility_20",
    "volatility_60",
    "avg_amount_liquidity",
    "amount_ratio_20",
    "volume_ratio_20",
    "price_position_20",
    "amihud_20",
    "gap_volnorm_20",
    "momentum_5_volnorm_20",
    "trend_5_20",
    "volatility_regime",
    "log_market_cap",
}

SAFE_FUNCTIONS = {
    "abs",
    "signed_log1p",
    "safe_div",
    "minimum",
    "maximum",
    "clip",
    "lag",
    "delta",
    "pct_change",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_sum",
    "ema",
    "ts_zscore",
}

WINDOW_FUNCTIONS = {
    "lag",
    "delta",
    "pct_change",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "rolling_sum",
    "ema",
    "ts_zscore",
}


class FeatureDefinitionError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    expression: str
    rationale: str
    expected_direction: str = "unknown"
    source: str = "deepseek"
    screening: dict[str, float | int | str | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.screening:
            payload.pop("screening")
        return payload


@dataclass(frozen=True)
class FeatureScreeningResult:
    accepted: list[FeatureDefinition]
    report: pd.DataFrame
    screening_start: str
    screening_end: str
    label_data_end: str
    screening_rows: int
    candidate_features: int


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def _constant_number(node: ast.AST, label: str) -> float:
    sign = 1.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        sign = -1.0 if isinstance(node.op, ast.USub) else 1.0
        node = node.operand
    if (
        not isinstance(node, ast.Constant)
        or isinstance(node.value, bool)
        or not isinstance(node.value, (int, float))
    ):
        raise FeatureDefinitionError(f"{label} must be a numeric constant")
    value = sign * float(node.value)
    if not math.isfinite(value) or abs(value) > 1e12:
        raise FeatureDefinitionError(f"{label} is outside the permitted range")
    return value


def _constant_window(node: ast.AST, function_name: str) -> int:
    value = _constant_number(node, f"{function_name} window")
    if not value.is_integer():
        raise FeatureDefinitionError(f"{function_name} window must be an integer")
    window = int(value)
    minimum = 1 if function_name in {"lag", "delta", "pct_change"} else 2
    if not minimum <= window <= MAX_LOOKBACK:
        raise FeatureDefinitionError(
            f"{function_name} window must be in [{minimum}, {MAX_LOOKBACK}]"
        )
    return window


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: set[str]):
        self.allowed_names = allowed_names
        self.referenced_names: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.UAdd,
            ast.USub,
        )
        if not isinstance(node, allowed_nodes):
            raise FeatureDefinitionError(
                f"Expression contains forbidden syntax: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names and node.id not in SAFE_FUNCTIONS:
            raise FeatureDefinitionError(f"Unknown or forbidden field: {node.id}")
        if node.id in self.allowed_names:
            self.referenced_names.add(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        _constant_number(node, "constant")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise FeatureDefinitionError("Only whitelisted feature functions may be called")
        if node.keywords:
            raise FeatureDefinitionError("Keyword arguments are not permitted")

        name = node.func.id
        expected_arguments = {
            "abs": 1,
            "signed_log1p": 1,
            "safe_div": 2,
            "minimum": 2,
            "maximum": 2,
            "clip": 3,
        }.get(name, 2)
        if len(node.args) != expected_arguments:
            raise FeatureDefinitionError(
                f"{name} requires exactly {expected_arguments} arguments"
            )
        if name in WINDOW_FUNCTIONS:
            _constant_window(node.args[1], name)
        if name == "clip":
            lower = _constant_number(node.args[1], "clip lower bound")
            upper = _constant_number(node.args[2], "clip upper bound")
            if lower >= upper:
                raise FeatureDefinitionError("clip lower bound must be below upper bound")
        self.visit(node.func)
        for argument in node.args:
            self.visit(argument)


def validate_expression(
    expression: str,
    allowed_names: Iterable[str],
) -> tuple[ast.Expression, set[str], str]:
    if not isinstance(expression, str) or not expression.strip():
        raise FeatureDefinitionError("Feature expression cannot be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise FeatureDefinitionError(
            f"Feature expression exceeds {MAX_EXPRESSION_LENGTH} characters"
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise FeatureDefinitionError(f"Invalid feature expression: {error.msg}") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise FeatureDefinitionError(
            f"Feature expression exceeds {MAX_AST_NODES} syntax nodes"
        )
    if _ast_depth(tree) > MAX_AST_DEPTH:
        raise FeatureDefinitionError(
            f"Feature expression exceeds maximum depth {MAX_AST_DEPTH}"
        )
    validator = _ExpressionValidator(set(allowed_names))
    validator.visit(tree)
    if not validator.referenced_names:
        raise FeatureDefinitionError("Feature expression must reference market data")
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return tree, validator.referenced_names, canonical


def _feature_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        return payload["features"]
    raise FeatureDefinitionError(
        "Feature file must be a list or an object containing a 'features' list"
    )


def parse_feature_candidates(
    payload: Any,
    allowed_names: Iterable[str] = DEFAULT_GENERATED_FEATURE_INPUTS,
) -> tuple[list[FeatureDefinition], list[dict[str, Any]]]:
    allowed = set(allowed_names)
    valid: list[FeatureDefinition] = []
    rejected: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_expressions: set[str] = set()

    for index, item in enumerate(_feature_items(payload)):
        try:
            if not isinstance(item, dict):
                raise FeatureDefinitionError("Feature entry must be a JSON object")
            name = str(item.get("name", "")).strip()
            expression = str(item.get("expression", "")).strip()
            rationale = str(item.get("rationale", "")).strip()
            expected_direction = str(
                item.get("expected_direction", "unknown")
            ).strip().lower()
            source = str(item.get("source", "deepseek")).strip() or "deepseek"

            if not FEATURE_NAME_PATTERN.fullmatch(name):
                raise FeatureDefinitionError(
                    "Feature name must match ai_[a-z][a-z0-9_]{2,59}"
                )
            if name in seen_names:
                raise FeatureDefinitionError(f"Duplicate feature name: {name}")
            if len(rationale) < 10 or len(rationale) > 600:
                raise FeatureDefinitionError(
                    "Feature rationale must contain 10 to 600 characters"
                )
            if expected_direction not in {"positive", "negative", "unknown"}:
                raise FeatureDefinitionError(
                    "expected_direction must be positive, negative, or unknown"
                )
            _, _, canonical = validate_expression(expression, allowed)
            if canonical in seen_expressions:
                raise FeatureDefinitionError("Duplicate feature expression")

            screening = item.get("screening", {})
            if not isinstance(screening, dict):
                raise FeatureDefinitionError("screening metadata must be an object")
            definition = FeatureDefinition(
                name=name,
                expression=expression,
                rationale=rationale,
                expected_direction=expected_direction,
                source=source,
                screening=dict(screening),
            )
            valid.append(definition)
            seen_names.add(name)
            seen_expressions.add(canonical)
        except (FeatureDefinitionError, TypeError, ValueError) as error:
            rejected.append(
                {
                    "index": index,
                    "name": item.get("name") if isinstance(item, dict) else None,
                    "reason": str(error),
                }
            )
    return valid, rejected


def load_feature_definitions(
    path: str | Path,
    allowed_names: Iterable[str] = DEFAULT_GENERATED_FEATURE_INPUTS,
) -> list[FeatureDefinition]:
    feature_path = Path(path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Generated feature file not found: {feature_path}")
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    valid, rejected = parse_feature_candidates(payload, allowed_names)
    if rejected:
        examples = "; ".join(
            f"#{item['index']} {item['reason']}" for item in rejected[:5]
        )
        raise FeatureDefinitionError(
            f"Generated feature file contains invalid entries: {examples}"
        )
    if not valid:
        raise FeatureDefinitionError("Generated feature file contains no features")
    return valid


class FeatureExpressionEvaluator:
    def __init__(self, frame: pd.DataFrame, allowed_names: Iterable[str]):
        self.frame = frame
        self.allowed_names = set(allowed_names)
        missing = sorted(self.allowed_names - set(frame.columns))
        if missing:
            raise FeatureDefinitionError(
                f"Generated feature inputs are missing from the frame: {missing[:10]}"
            )

    def _series(self, value: Any, function_name: str) -> pd.Series:
        if not isinstance(value, pd.Series):
            raise FeatureDefinitionError(
                f"{function_name} first argument must resolve to a data series"
            )
        return pd.to_numeric(value, errors="coerce")

    def _group(self, series: pd.Series) -> pd.core.groupby.SeriesGroupBy:
        return series.groupby(self.frame["code"], sort=False)

    @staticmethod
    def _clean(value: Any) -> Any:
        if isinstance(value, pd.Series):
            return value.replace([np.inf, -np.inf], np.nan)
        if isinstance(value, np.ndarray):
            return np.where(np.isfinite(value), value, np.nan)
        return value

    def _binary(self, node: ast.BinOp) -> Any:
        left = self._evaluate_node(node.left)
        right = self._evaluate_node(node.right)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                result = left / right
            else:  # pragma: no cover - validator prevents this branch.
                raise FeatureDefinitionError("Unsupported arithmetic operator")
        return self._clean(result)

    def _evaluate_call(self, node: ast.Call) -> Any:
        name = node.func.id
        arguments = [self._evaluate_node(argument) for argument in node.args]

        if name == "abs":
            return np.abs(arguments[0])
        if name == "signed_log1p":
            value = arguments[0]
            return np.sign(value) * np.log1p(np.abs(value))
        if name == "safe_div":
            numerator, denominator = arguments
            if isinstance(denominator, pd.Series):
                safe_denominator = denominator.where(denominator.abs() > 1e-12)
            else:
                safe_denominator = (
                    np.nan if abs(float(denominator)) <= 1e-12 else denominator
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                return self._clean(numerator / safe_denominator)
        if name in {"minimum", "maximum"}:
            function = np.minimum if name == "minimum" else np.maximum
            result = function(arguments[0], arguments[1])
            if isinstance(result, np.ndarray):
                result = pd.Series(result, index=self.frame.index)
            return self._clean(result)
        if name == "clip":
            value = arguments[0]
            lower = float(arguments[1])
            upper = float(arguments[2])
            if isinstance(value, pd.Series):
                return value.clip(lower=lower, upper=upper)
            return float(np.clip(value, lower, upper))

        series = self._series(arguments[0], name)
        window = int(arguments[1])
        grouped = self._group(series)
        if name == "lag":
            return grouped.shift(window)
        if name == "delta":
            return series - grouped.shift(window)
        if name == "pct_change":
            previous = grouped.shift(window)
            return self._clean(series / previous.where(previous.abs() > 1e-12) - 1.0)

        def rolling(values: pd.Series, operation: str) -> pd.Series:
            windowed = values.rolling(window, min_periods=window)
            if operation == "mean":
                return windowed.mean()
            if operation == "std":
                return windowed.std(ddof=0)
            if operation == "min":
                return windowed.min()
            if operation == "max":
                return windowed.max()
            if operation == "sum":
                return windowed.sum()
            raise FeatureDefinitionError(f"Unsupported rolling operation: {operation}")

        if name.startswith("rolling_"):
            operation = name.removeprefix("rolling_")
            return grouped.transform(lambda values: rolling(values, operation))
        if name == "ema":
            return grouped.transform(
                lambda values: values.ewm(
                    span=window,
                    min_periods=window,
                    adjust=False,
                ).mean()
            )
        if name == "ts_zscore":
            mean = grouped.transform(lambda values: rolling(values, "mean"))
            std = grouped.transform(lambda values: rolling(values, "std"))
            return self._clean((series - mean) / std.where(std > 1e-12))
        raise FeatureDefinitionError(f"Unsupported feature function: {name}")

    def _evaluate_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            return self.frame[node.id]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            return self._binary(node)
        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Call):
            return self._evaluate_call(node)
        raise FeatureDefinitionError(
            f"Unsupported expression node: {type(node).__name__}"
        )

    def evaluate(self, expression: str) -> pd.Series:
        tree, _, _ = validate_expression(expression, self.allowed_names)
        result = self._evaluate_node(tree.body)
        if not isinstance(result, pd.Series):
            raise FeatureDefinitionError(
                "Feature expression must produce one value per market-data row"
            )
        if len(result) != len(self.frame):
            raise FeatureDefinitionError("Generated feature has an invalid row count")
        return pd.to_numeric(result, errors="coerce").replace(
            [np.inf, -np.inf],
            np.nan,
        )


def apply_generated_features(
    frame: pd.DataFrame,
    definitions: Sequence[FeatureDefinition],
    allowed_names: Iterable[str],
) -> list[str]:
    allowed = set(allowed_names)
    evaluator = FeatureExpressionEvaluator(frame, allowed)
    generated_columns: list[str] = []
    for definition in definitions:
        if definition.name in frame.columns:
            raise FeatureDefinitionError(
                f"Generated feature collides with an existing column: {definition.name}"
            )
        frame[definition.name] = evaluator.evaluate(definition.expression)
        frame[definition.name] = frame[definition.name].astype(np.float32)
        generated_columns.append(definition.name)
    return generated_columns


def _daily_information_coefficients(
    values: pd.DataFrame,
    feature_column: str,
    min_cross_sectional_stocks: int,
) -> pd.Series:
    records: dict[pd.Timestamp, float] = {}
    for date, group in values.groupby("date", sort=False):
        valid = group[[feature_column, "target_rank"]].dropna()
        if len(valid) < min_cross_sectional_stocks:
            continue
        if (
            valid[feature_column].nunique(dropna=True) < 2
            or valid["target_rank"].nunique(dropna=True) < 2
        ):
            continue
        correlation = valid[feature_column].corr(valid["target_rank"])
        if pd.notna(correlation):
            records[pd.Timestamp(date)] = float(correlation)
    return pd.Series(records, dtype=float).sort_index()


def _newey_west_mean_tstat(values: pd.Series, max_lag: int) -> float:
    observations = values.dropna().to_numpy(dtype=float)
    count = len(observations)
    if count < 2:
        return math.nan
    centered = observations - observations.mean()
    long_run_variance = float(np.dot(centered, centered) / count)
    lag_limit = min(max(0, int(max_lag)), count - 1)
    for lag in range(1, lag_limit + 1):
        covariance = float(
            np.dot(centered[lag:], centered[:-lag]) / count
        )
        weight = 1.0 - lag / (lag_limit + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if not math.isfinite(long_run_variance) or long_run_variance <= 0:
        return math.nan
    standard_error = math.sqrt(long_run_variance / count)
    return float(observations.mean() / standard_error)


def screen_feature_definitions(
    market_data: pd.DataFrame,
    config: AppConfig,
    definitions: Sequence[FeatureDefinition],
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> FeatureScreeningResult:
    if not definitions:
        raise FeatureDefinitionError("No generated feature definitions to screen")

    from .features import build_features

    prepared = build_features(
        market_data,
        config,
        generated_definitions=list(definitions),
    )
    frame = prepared.frame
    mask = frame["eligible"] & frame["target_rank"].notna()
    if start_date is not None:
        mask &= frame["date"].ge(pd.Timestamp(start_date))
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"].unique())))
    if end_date is not None:
        requested_end = pd.Timestamp(end_date)
        label_dates = calendar[calendar <= requested_end]
        if len(label_dates) <= config.features.prediction_horizon + 1:
            raise ValueError(
                "Screening end date leaves too little history for forward labels"
            )
        label_data_end = pd.Timestamp(label_dates[-1])
        safe_feature_end_index = (
            int(calendar.get_loc(label_data_end))
            - config.features.prediction_horizon
            - 1
        )
        safe_feature_end = pd.Timestamp(calendar[safe_feature_end_index])
        mask &= frame["date"].le(safe_feature_end)
    else:
        label_data_end = pd.Timestamp(calendar[-1])
    generated_columns = [f"{definition.name}_cs" for definition in definitions]
    values = frame.loc[
        mask,
        ["date", "target_rank", *generated_columns],
    ].copy()
    if values.empty:
        raise ValueError("No eligible labeled rows in the requested screening period")

    screening = config.feature_screening
    report_records: list[dict[str, Any]] = []
    preliminary: list[str] = []
    metrics_by_name: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        column = f"{definition.name}_cs"
        coverage = float(values[column].notna().mean())
        daily_ic = _daily_information_coefficients(
            values,
            column,
            screening.min_cross_sectional_stocks,
        )
        ic_days = len(daily_ic)
        mean_ic = float(daily_ic.mean()) if ic_days else math.nan
        ic_std = float(daily_ic.std(ddof=1)) if ic_days > 1 else math.nan
        ic_tstat = _newey_west_mean_tstat(
            daily_ic,
            max_lag=config.features.prediction_horizon,
        )
        positive_ic_fraction = (
            float(daily_ic.gt(0).mean()) if ic_days else math.nan
        )

        rejection_reasons: list[str] = []
        if coverage < screening.min_feature_coverage:
            rejection_reasons.append("coverage")
        if ic_days < screening.min_screening_days:
            rejection_reasons.append("screening_days")
        if not math.isfinite(mean_ic) or abs(mean_ic) < screening.min_abs_mean_ic:
            rejection_reasons.append("mean_ic")
        if not math.isfinite(ic_tstat) or abs(ic_tstat) < screening.min_abs_ic_tstat:
            rejection_reasons.append("ic_tstat")

        metrics = {
            "coverage": coverage,
            "ic_days": ic_days,
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_tstat": ic_tstat,
            "positive_ic_fraction": positive_ic_fraction,
            "ic_tstat_method": "newey_west",
            "ic_hac_lags": config.features.prediction_horizon,
        }
        metrics_by_name[definition.name] = metrics
        if not rejection_reasons:
            preliminary.append(definition.name)
        report_records.append(
            {
                "name": definition.name,
                "expression": definition.expression,
                "rationale": definition.rationale,
                "expected_direction": definition.expected_direction,
                **metrics,
                "status": (
                    "preliminary"
                    if not rejection_reasons
                    else "rejected_threshold"
                ),
                "rejection_reason": ",".join(rejection_reasons),
                "max_abs_correlation": math.nan,
                "correlated_with": "",
            }
        )

    ordered = sorted(
        preliminary,
        key=lambda name: abs(float(metrics_by_name[name]["mean_ic"])),
        reverse=True,
    )
    feature_columns = [f"{name}_cs" for name in ordered]
    sample = values[feature_columns]
    if len(sample) > screening.correlation_sample_rows:
        sample = sample.sample(
            screening.correlation_sample_rows,
            random_state=42,
        )
    correlations = sample.corr().abs() if feature_columns else pd.DataFrame()

    accepted_names: list[str] = []
    redundancy: dict[str, tuple[float, str, str]] = {}
    for name in ordered:
        if len(accepted_names) >= screening.max_selected_features:
            redundancy[name] = (math.nan, "", "max_selected_features")
            continue
        column = f"{name}_cs"
        best_correlation = 0.0
        best_name = ""
        for accepted_name in accepted_names:
            other = f"{accepted_name}_cs"
            correlation = correlations.at[column, other]
            if pd.notna(correlation) and float(correlation) > best_correlation:
                best_correlation = float(correlation)
                best_name = accepted_name
        if best_correlation > screening.max_pairwise_correlation:
            redundancy[name] = (
                best_correlation,
                best_name,
                "pairwise_correlation",
            )
            continue
        accepted_names.append(name)
        redundancy[name] = (best_correlation, best_name, "")

    report = pd.DataFrame(report_records)
    for index, row in report.iterrows():
        name = str(row["name"])
        if name not in preliminary:
            continue
        correlation, correlated_with, rejection = redundancy[name]
        report.at[index, "max_abs_correlation"] = correlation
        report.at[index, "correlated_with"] = correlated_with
        if rejection:
            report.at[index, "status"] = "rejected_redundancy"
            report.at[index, "rejection_reason"] = rejection
        else:
            report.at[index, "status"] = "accepted"

    definition_by_name = {definition.name: definition for definition in definitions}
    accepted = [
        replace(
            definition_by_name[name],
            screening={
                key: (
                    None
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in metrics_by_name[name].items()
            },
        )
        for name in accepted_names
    ]
    minimum_date = pd.Timestamp(values["date"].min()).date().isoformat()
    maximum_date = pd.Timestamp(values["date"].max()).date().isoformat()
    return FeatureScreeningResult(
        accepted=accepted,
        report=report.sort_values(
            ["status", "mean_ic"],
            ascending=[True, False],
            kind="stable",
        ).reset_index(drop=True),
        screening_start=minimum_date,
        screening_end=maximum_date,
        label_data_end=label_data_end.date().isoformat(),
        screening_rows=len(values),
        candidate_features=len(definitions),
    )


def write_feature_screening_result(
    result: FeatureScreeningResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    accepted_path = directory / "accepted_features.json"
    report_path = directory / "screening_report.csv"
    metadata_path = directory / "screening_summary.json"

    accepted_payload = {
        "schema_version": FEATURE_FILE_SCHEMA_VERSION,
        "screening": {
            "start": result.screening_start,
            "end": result.screening_end,
            "label_data_end": result.label_data_end,
            "rows": result.screening_rows,
            "candidate_features": result.candidate_features,
            "accepted_features": len(result.accepted),
        },
        "features": [definition.to_dict() for definition in result.accepted],
    }
    accepted_path.write_text(
        json.dumps(accepted_payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    result.report.to_csv(report_path, index=False)
    metadata_path.write_text(
        json.dumps(
            accepted_payload["screening"],
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "accepted_features": accepted_path,
        "screening_report": report_path,
        "screening_summary": metadata_path,
    }
