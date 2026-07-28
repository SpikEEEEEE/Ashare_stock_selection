from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ashare_selection.config import AppConfig
from ashare_selection.data import load_market_data
from ashare_selection.demo import generate_demo_data
from ashare_selection.features import build_features
from ashare_selection.pipeline import CandidateSelector


def small_test_config() -> AppConfig:
    config = AppConfig()
    config.universe.min_listing_days = 20
    config.universe.min_avg_amount = 0.0
    config.features.min_feature_history = 20
    config.features.prediction_horizon = 5
    config.model.train_lookback_days = 160
    config.model.min_train_days = 100
    config.model.min_train_rows = 1_000
    config.selection.top_k = 10
    config.selection.max_industry_fraction = 0.30
    config.backtest.rebalance_every_days = 5
    config.backtest.max_periods = 3
    return config


class CandidatePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_path = Path(cls.temp_dir.name) / "market.csv"
        generate_demo_data(cls.data_path, stocks=40, days=360, seed=11)
        cls.config = small_test_config()
        cls.market = load_market_data(cls.data_path, cls.config)
        cls.prepared = build_features(cls.market, cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def test_features_do_not_change_when_future_rows_are_removed(self) -> None:
        all_dates = sorted(self.market["date"].unique())
        cutoff = all_dates[-30]
        truncated_market = self.market.loc[self.market["date"].le(cutoff)].copy()
        truncated = build_features(truncated_market, self.config)

        full_row = self.prepared.frame.loc[
            self.prepared.frame["date"].eq(cutoff),
            ["code", *self.prepared.feature_columns],
        ].sort_values("code")
        truncated_row = truncated.frame.loc[
            truncated.frame["date"].eq(cutoff),
            ["code", *truncated.feature_columns],
        ].sort_values("code")
        self.assertEqual(list(full_row["code"]), list(truncated_row["code"]))
        np.testing.assert_allclose(
            full_row[self.prepared.feature_columns].to_numpy(float),
            truncated_row[truncated.feature_columns].to_numpy(float),
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_selection_respects_filters_and_industry_cap(self) -> None:
        selector = CandidateSelector(self.config)
        result = selector.select_prepared(self.prepared)
        candidates = result.candidates
        self.assertEqual(len(candidates), self.config.selection.top_k)
        self.assertTrue(candidates["eligible"].all())
        self.assertFalse(candidates["is_st"].any())
        self.assertFalse(candidates["is_suspended"].any())
        self.assertFalse(candidates["is_limit_up"].any())
        cap = int(
            np.ceil(
                self.config.selection.top_k
                * self.config.selection.max_industry_fraction
            )
        )
        self.assertLessEqual(int(candidates["industry"].value_counts().max()), cap)

    def test_walk_forward_backtest_runs(self) -> None:
        selector = CandidateSelector(self.config)
        result = selector.backtest_prepared(self.prepared)
        self.assertEqual(len(result.periods), 3)
        self.assertEqual(
            len(result.holdings), 3 * self.config.selection.top_k
        )
        self.assertIn("mean_spearman_ic", result.summary)


if __name__ == "__main__":
    unittest.main()

