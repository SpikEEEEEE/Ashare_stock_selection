from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_selection.config import AppConfig
from ashare_selection.pipeline import CandidateSelector
from ashare_selection.tushare_source import TushareDataSource


class FakeTushareClient:
    def __init__(self, stocks: int = 25, days: int = 140):
        self.dates = pd.bdate_range("2025-01-02", periods=days)
        self.codes = [
            f"{index + 1:06d}.SZ" if index % 2 == 0 else f"{600000 + index:06d}.SH"
            for index in range(stocks)
        ]
        self.calls: Counter[str] = Counter()
        self.date_to_index = {
            date.strftime("%Y%m%d"): index for index, date in enumerate(self.dates)
        }

    def trade_cal(self, **kwargs: object) -> pd.DataFrame:
        self.calls["trade_cal"] += 1
        start = pd.Timestamp(str(kwargs["start_date"]))
        end = pd.Timestamp(str(kwargs["end_date"]))
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame(
            {
                "exchange": "SSE",
                "cal_date": dates.strftime("%Y%m%d"),
                "is_open": dates.weekday < 5,
                "pretrade_date": "",
            }
        )

    def stock_basic(self, **kwargs: object) -> pd.DataFrame:
        self.calls["stock_basic"] += 1
        if kwargs.get("list_status") != "L":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "symbol": [code.split(".")[0] for code in self.codes],
                "name": [
                    "ST测试一" if index == 0 else f"测试{index}"
                    for index in range(len(self.codes))
                ],
                "area": "测试",
                "industry": [
                    f"行业{index % 5}" for index in range(len(self.codes))
                ],
                "market": "主板",
                "exchange": [
                    "SZSE" if code.endswith(".SZ") else "SSE" for code in self.codes
                ],
                "list_status": "L",
                "list_date": "20200101",
                "delist_date": pd.NA,
            }
        )

    def namechange(self, **_kwargs: object) -> pd.DataFrame:
        self.calls["namechange"] += 1
        start = self.dates[-20].strftime("%Y%m%d")
        return pd.DataFrame(
            {
                "ts_code": [self.codes[0]],
                "name": ["ST测试一"],
                "start_date": [start],
                "end_date": [pd.NA],
                "change_reason": ["ST"],
            }
        )

    def daily(self, **kwargs: object) -> pd.DataFrame:
        self.calls["daily"] += 1
        trade_date = str(kwargs["trade_date"])
        day = self.date_to_index[trade_date]
        stock_index = np.arange(len(self.codes), dtype=float)
        close = 8.0 + stock_index * 0.15 + day * 0.003
        open_price = close * (1.0 - 0.001 * ((stock_index % 3) - 1))
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "trade_date": trade_date,
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.01,
                "low": np.minimum(open_price, close) * 0.99,
                "close": close,
                "pre_close": close - 0.003,
                "pct_chg": 0.03,
                "vol": 100_000.0 + stock_index * 1_000.0,
                "amount": 10_000.0 + stock_index * 100.0,
            }
        )

    def daily_basic(self, **kwargs: object) -> pd.DataFrame:
        self.calls["daily_basic"] += 1
        trade_date = str(kwargs["trade_date"])
        day = self.date_to_index[trade_date]
        limit_status = np.zeros(len(self.codes), dtype=int)
        if day == len(self.dates) - 1:
            limit_status[1] = 2
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "trade_date": trade_date,
                "turnover_rate": 1.0 + np.arange(len(self.codes)) * 0.01,
                "total_mv": 500_000.0 + np.arange(len(self.codes)) * 10_000.0,
                "circ_mv": 400_000.0 + np.arange(len(self.codes)) * 8_000.0,
                "limit_status": limit_status,
            }
        )

    def adj_factor(self, **kwargs: object) -> pd.DataFrame:
        self.calls["adj_factor"] += 1
        trade_date = str(kwargs["trade_date"])
        return pd.DataFrame(
            {
                "ts_code": self.codes,
                "trade_date": trade_date,
                "adj_factor": 1.0,
            }
        )


def tushare_test_config(cache_dir: Path) -> AppConfig:
    config = AppConfig()
    config.tushare.cache_dir = str(cache_dir)
    config.tushare.request_interval_seconds = 0.0
    config.tushare.max_retries = 1
    config.tushare.refresh_last_trading_days = 0
    config.universe.min_listing_days = 20
    config.universe.min_avg_amount = 0.0
    config.features.min_feature_history = 20
    config.model.train_lookback_days = 100
    config.model.min_train_days = 60
    config.model.min_train_rows = 1_000
    config.selection.top_k = 5
    config.selection.max_industry_fraction = 0.4
    return config


class TushareSourceTest(unittest.TestCase):
    def test_download_cache_conversion_and_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeTushareClient()
            config = tushare_test_config(Path(temp_dir) / "cache")
            source = TushareDataSource(
                config, client=client, sleep=lambda _seconds: None
            )
            start = client.dates[0].strftime("%Y%m%d")
            end = client.dates[-1].strftime("%Y%m%d")

            market, stats = source.download(start, end)
            self.assertEqual(stats.trading_dates, len(client.dates))
            self.assertEqual(stats.output_rows, len(client.dates) * len(client.codes))
            self.assertEqual(market.iloc[0]["volume"], 100_000.0 * 100.0)
            self.assertEqual(market.iloc[0]["amount"], 10_000.0 * 1_000.0)
            self.assertEqual(market.iloc[0]["market_cap"], 500_000.0 * 10_000.0)
            self.assertAlmostEqual(market.iloc[0]["turnover_rate"], 0.01)

            first_call_counts = client.calls.copy()
            cached_market, cached_stats = source.download(start, end)
            self.assertEqual(client.calls, first_call_counts)
            self.assertEqual(len(cached_market), len(market))
            self.assertEqual(cached_stats.daily_cached, len(client.dates))

            selector = CandidateSelector(config)
            result = selector.select_prepared(selector.prepare(market))
            self.assertEqual(len(result.candidates), config.selection.top_k)
            self.assertNotIn(
                client.codes[0].split(".")[0],
                set(result.candidates["code"].astype(str)),
            )
            self.assertNotIn(
                client.codes[1].split(".")[0],
                set(result.candidates["code"].astype(str)),
            )


if __name__ == "__main__":
    unittest.main()

