from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ashare_selection.config import AppConfig
from ashare_selection.data import load_market_data
from ashare_selection.deepseek_features import DeepSeekFeatureGenerator
from ashare_selection.demo import generate_demo_data
from ashare_selection.features import build_features
from ashare_selection.generated_features import (
    FeatureDefinition,
    parse_feature_candidates,
    screen_feature_definitions,
    write_feature_screening_result,
)
from ashare_selection.pipeline import CandidateSelector


def generated_test_config() -> AppConfig:
    config = AppConfig()
    config.universe.min_listing_days = 20
    config.universe.min_avg_amount = 0.0
    config.features.min_feature_history = 20
    config.model.train_lookback_days = 160
    config.model.min_train_days = 100
    config.model.min_train_rows = 1_000
    config.model.n_estimators = 15
    config.model.num_leaves = 7
    config.model.min_child_samples = 10
    config.model.n_jobs = 1
    config.feature_screening.min_feature_coverage = 0.70
    config.feature_screening.min_screening_days = 20
    config.feature_screening.min_cross_sectional_stocks = 20
    config.feature_screening.min_abs_mean_ic = 0.0
    config.feature_screening.min_abs_ic_tstat = 0.0
    config.feature_screening.max_pairwise_correlation = 0.80
    config.feature_screening.correlation_sample_rows = 5_000
    config.feature_screening.max_selected_features = 5
    return config


class GeneratedFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.market_path = Path(cls.temp_dir.name) / "market.csv"
        generate_demo_data(cls.market_path, stocks=40, days=360, seed=29)
        cls.config = generated_test_config()
        cls.market = load_market_data(cls.market_path, cls.config)
        cls.definitions = [
            FeatureDefinition(
                name="ai_vol_adjusted_trend",
                expression="safe_div(momentum_5 - momentum_20, volatility_20)",
                rationale="短周期趋势相对中周期趋势走强，并用近期波动率归一化。",
                expected_direction="positive",
            ),
            FeatureDefinition(
                name="ai_vol_adjusted_trend_copy",
                expression=(
                    "safe_div(momentum_5 - momentum_20, volatility_20) * 1.0"
                ),
                rationale="用于验证高度相关的重复特征会在本地筛选阶段被去除。",
                expected_direction="positive",
            ),
            FeatureDefinition(
                name="ai_amount_ratio_zscore",
                expression="ts_zscore(amount_ratio_20, 20)",
                rationale="识别个股成交额相对自身近期状态的异常扩张或收缩。",
                expected_direction="unknown",
            ),
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_generated_features_are_point_in_time_and_join_model(self) -> None:
        prepared = build_features(
            self.market,
            self.config,
            generated_definitions=self.definitions,
        )
        self.assertIn("ai_vol_adjusted_trend_cs", prepared.feature_columns)
        self.assertIn("ai_amount_ratio_zscore_cs", prepared.feature_columns)

        cutoff = sorted(self.market["date"].unique())[-20]
        truncated = build_features(
            self.market.loc[self.market["date"].le(cutoff)].copy(),
            self.config,
            generated_definitions=self.definitions,
        )
        full = prepared.frame.loc[
            prepared.frame["date"].eq(cutoff),
            ["code", "ai_vol_adjusted_trend_cs", "ai_amount_ratio_zscore_cs"],
        ].sort_values("code")
        short = truncated.frame.loc[
            truncated.frame["date"].eq(cutoff),
            ["code", "ai_vol_adjusted_trend_cs", "ai_amount_ratio_zscore_cs"],
        ].sort_values("code")
        pd.testing.assert_frame_equal(
            full.reset_index(drop=True),
            short.reset_index(drop=True),
        )

    def test_unsafe_or_forward_expressions_are_rejected(self) -> None:
        payload = {
            "features": [
                {
                    "name": "ai_import_attack",
                    "expression": "__import__('os').system('whoami')",
                    "rationale": "This deliberately unsafe expression must be rejected locally.",
                },
                {
                    "name": "ai_future_price",
                    "expression": "lag(adj_close, -1)",
                    "rationale": "This deliberately forward-looking expression must be rejected.",
                },
            ]
        }
        valid, rejected = parse_feature_candidates(payload)
        self.assertEqual(valid, [])
        self.assertEqual(len(rejected), 2)

    def test_screening_removes_redundant_features_and_writes_outputs(self) -> None:
        dates = sorted(self.market["date"].unique())
        label_data_end = pd.Timestamp(dates[-10])
        safe_feature_end = pd.Timestamp(
            dates[-10 - self.config.features.prediction_horizon - 1]
        )
        result = screen_feature_definitions(
            self.market,
            self.config,
            self.definitions,
            end_date=label_data_end,
        )
        self.assertEqual(
            result.label_data_end,
            label_data_end.date().isoformat(),
        )
        self.assertLessEqual(
            pd.Timestamp(result.screening_end),
            safe_feature_end,
        )
        statuses = result.report.set_index("name")["status"].to_dict()
        duplicate_statuses = {
            statuses["ai_vol_adjusted_trend"],
            statuses["ai_vol_adjusted_trend_copy"],
        }
        self.assertIn("accepted", duplicate_statuses)
        self.assertIn("rejected_redundancy", duplicate_statuses)
        paths = write_feature_screening_result(
            result,
            Path(self.temp_dir.name) / "screened",
        )
        self.assertTrue(paths["accepted_features"].exists())
        self.assertTrue(paths["screening_report"].exists())
        self.config.features.generated_feature_path = str(paths["accepted_features"])
        try:
            selector = CandidateSelector(self.config)
            selection = selector.select_prepared(selector.prepare(self.market))
            self.assertGreater(selection.diagnostics.active_features, 20)
        finally:
            self.config.features.generated_feature_path = None


class DeepSeekClientTest(unittest.TestCase):
    def test_json_response_is_validated_before_becoming_features(self) -> None:
        response_payload = {
            "id": "response-test",
            "model": "deepseek-v4-pro",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "features": [
                                    {
                                        "name": "ai_valid_interaction",
                                        "expression": (
                                            "safe_div(momentum_5, volatility_20)"
                                        ),
                                        "rationale": (
                                            "Risk-adjusted short-term momentum may "
                                            "separate persistent moves from noise."
                                        ),
                                        "expected_direction": "positive",
                                    },
                                    {
                                        "name": "ai_invalid_future",
                                        "expression": "lag(adj_close, -5)",
                                        "rationale": (
                                            "This future-looking proposal must be rejected."
                                        ),
                                    },
                                ]
                            }
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        captured_request: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(response_payload).encode("utf-8")

        def fake_opener(request: object, timeout: float) -> FakeResponse:
            captured_request["payload"] = json.loads(request.data.decode("utf-8"))
            captured_request["timeout"] = timeout
            return FakeResponse()

        config = AppConfig()
        config.deepseek.max_retries = 1
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            result = DeepSeekFeatureGenerator(
                config,
                opener=fake_opener,
                sleep=lambda _seconds: None,
            ).generate(2)

        self.assertEqual(len(result.features), 1)
        self.assertEqual(len(result.rejected), 1)
        request_payload = captured_request["payload"]
        self.assertEqual(request_payload["response_format"], {"type": "json_object"})
        self.assertEqual(request_payload["model"], "deepseek-v4-pro")


if __name__ == "__main__":
    unittest.main()
