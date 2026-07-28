from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import AppConfig


ProgressCallback = Callable[[str], None]


@dataclass
class TushareDownloadStats:
    trading_dates: int = 0
    daily_downloaded: int = 0
    daily_cached: int = 0
    daily_basic_downloaded: int = 0
    daily_basic_cached: int = 0
    adj_factor_downloaded: int = 0
    adj_factor_cached: int = 0
    output_rows: int = 0
    latest_trade_date: str | None = None
    warnings: list[str] = field(default_factory=list)


def create_tushare_client(config: AppConfig) -> Any:
    token_name = config.tushare.token_env
    token = os.environ.get(token_name, "").strip()
    if not token:
        raise RuntimeError(
            f"Tushare token is missing. Set it first with: "
            f"export {token_name}='your-token'"
        )
    try:
        import tushare as ts
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The 'tushare' package is not installed. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from error

    ts.set_token(token)
    return ts.pro_api(token)


class TushareDataSource:
    """Incremental, date-partitioned Tushare Pro downloader."""

    DAILY_FIELDS = (
        "ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount"
    )
    DAILY_BASIC_FIELDS = (
        "ts_code,trade_date,turnover_rate,total_mv,circ_mv,limit_status"
    )
    ADJ_FACTOR_FIELDS = "ts_code,trade_date,adj_factor"
    STOCK_BASIC_FIELDS = (
        "ts_code,symbol,name,area,industry,market,exchange,"
        "list_status,list_date,delist_date"
    )
    NAMECHANGE_FIELDS = "ts_code,name,start_date,end_date,change_reason"

    def __init__(
        self,
        config: AppConfig,
        client: Any | None = None,
        progress: ProgressCallback | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.client = client if client is not None else create_tushare_client(config)
        self.cache_dir = Path(config.tushare.cache_dir)
        self.progress = progress or (lambda _message: None)
        self.sleep = sleep
        self._last_request_time = 0.0

    @staticmethod
    def _date_string(value: str | pd.Timestamp) -> str:
        parsed = pd.Timestamp(value)
        return parsed.strftime("%Y%m%d")

    def _respect_rate_limit(self) -> None:
        interval = self.config.tushare.request_interval_seconds
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < interval:
            self.sleep(interval - elapsed)

    def _call(self, endpoint: str, **kwargs: Any) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.config.tushare.max_retries + 1):
            try:
                self._respect_rate_limit()
                result = getattr(self.client, endpoint)(**kwargs)
                self._last_request_time = time.monotonic()
                if result is None:
                    return pd.DataFrame()
                if not isinstance(result, pd.DataFrame):
                    return pd.DataFrame(result)
                return result
            except Exception as error:  # API packages expose multiple error types.
                last_error = error
                if attempt >= self.config.tushare.max_retries:
                    break
                wait_seconds = (
                    self.config.tushare.retry_backoff_seconds * attempt
                )
                self.progress(
                    f"{endpoint} 调用失败，第 {attempt} 次重试，"
                    f"{wait_seconds:.1f} 秒后继续：{error}"
                )
                self.sleep(wait_seconds)
        raise RuntimeError(
            f"Tushare endpoint '{endpoint}' failed after "
            f"{self.config.tushare.max_retries} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(
            path,
            dtype={
                "ts_code": "string",
                "trade_date": "string",
                "cal_date": "string",
                "symbol": "string",
                "list_date": "string",
                "delist_date": "string",
                "start_date": "string",
                "end_date": "string",
            },
            low_memory=False,
        )

    @staticmethod
    def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)

    def _cache_path(self, endpoint: str, trade_date: str) -> Path:
        return self.cache_dir / endpoint / f"{trade_date}.csv"

    def _master_is_fresh(self, path: Path, max_age_days: int) -> bool:
        if not path.exists():
            return False
        age_seconds = time.time() - path.stat().st_mtime
        return age_seconds <= max_age_days * 86_400

    def fetch_stock_basic(self, force_refresh: bool = False) -> pd.DataFrame:
        path = self.cache_dir / "master" / "stock_basic.csv"
        if not force_refresh and self._master_is_fresh(path, 7):
            return self._read_csv(path)

        frames: list[pd.DataFrame] = []
        for status in ("L", "D", "P"):
            result = self._call(
                "stock_basic",
                exchange="",
                list_status=status,
                fields=self.STOCK_BASIC_FIELDS,
            )
            if not result.empty:
                frames.append(result)
        if not frames:
            raise RuntimeError("Tushare stock_basic returned no records")
        stocks = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("ts_code", keep="first")
            .reset_index(drop=True)
        )
        self._atomic_write(stocks, path)
        return stocks

    def fetch_name_history(
        self,
        force_refresh: bool = False,
        warnings: list[str] | None = None,
    ) -> pd.DataFrame:
        if not self.config.tushare.include_name_history:
            return pd.DataFrame()
        path = self.cache_dir / "master" / "namechange.csv"
        if not force_refresh and self._master_is_fresh(path, 30):
            return self._read_csv(path)
        try:
            history = self._call("namechange", fields=self.NAMECHANGE_FIELDS)
        except RuntimeError as error:
            message = (
                "无法取得 namechange 历史名称，将仅使用当前股票名称识别 ST："
                f"{error}"
            )
            if warnings is not None:
                warnings.append(message)
            self.progress(message)
            return pd.DataFrame()
        self._atomic_write(history, path)
        return history

    def fetch_trade_dates(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        force_refresh: bool = False,
    ) -> list[str]:
        start = self._date_string(start_date)
        end = self._date_string(end_date)
        path = self.cache_dir / "master" / "trade_cal.csv"
        calendar = pd.DataFrame()
        if path.exists() and not force_refresh:
            calendar = self._read_csv(path)
            cached_dates = pd.to_datetime(
                calendar.get("cal_date", pd.Series(dtype="string")),
                errors="coerce",
            )
            covers_range = (
                cached_dates.notna().any()
                and cached_dates.min() <= pd.Timestamp(start)
                and cached_dates.max() >= pd.Timestamp(end)
            )
            if not covers_range:
                calendar = pd.DataFrame()

        if calendar.empty:
            calendar = self._call(
                "trade_cal",
                exchange="SSE",
                start_date=start,
                end_date=end,
                is_open="",
            )
            if calendar.empty:
                raise RuntimeError("Tushare trade_cal returned no records")
            self._atomic_write(calendar, path)

        calendar["cal_date"] = calendar["cal_date"].astype("string")
        open_mask = pd.to_numeric(calendar["is_open"], errors="coerce").eq(1)
        within_range = calendar["cal_date"].between(start, end)
        return sorted(calendar.loc[open_mask & within_range, "cal_date"].tolist())

    def _download_endpoint_for_date(
        self,
        endpoint: str,
        fields: str,
        trade_date: str,
        refresh: bool,
        stats: TushareDownloadStats,
    ) -> pd.DataFrame:
        path = self._cache_path(endpoint, trade_date)
        cached_counter = f"{endpoint}_cached"
        downloaded_counter = f"{endpoint}_downloaded"
        if path.exists() and not refresh:
            setattr(stats, cached_counter, getattr(stats, cached_counter) + 1)
            return self._read_csv(path)

        result = self._call(endpoint, trade_date=trade_date, fields=fields)
        if result.empty:
            return result
        result["trade_date"] = result["trade_date"].astype("string")
        unexpected_dates = set(result["trade_date"].dropna().unique()) - {trade_date}
        if unexpected_dates:
            raise RuntimeError(
                f"{endpoint} returned unexpected dates for {trade_date}: "
                f"{sorted(unexpected_dates)[:5]}"
            )
        self._atomic_write(result, path)
        setattr(stats, downloaded_counter, getattr(stats, downloaded_counter) + 1)
        return result

    def download(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        force_refresh: bool = False,
        refresh_master: bool = False,
    ) -> tuple[pd.DataFrame, TushareDownloadStats]:
        stats = TushareDownloadStats()
        trade_dates = self.fetch_trade_dates(
            start_date, end_date, force_refresh=refresh_master
        )
        stats.trading_dates = len(trade_dates)
        if not trade_dates:
            raise RuntimeError("No open trading dates in the requested period")

        refresh_count = self.config.tushare.refresh_last_trading_days
        automatically_refreshed = set(
            trade_dates[-refresh_count:] if refresh_count else []
        )

        usable_dates: list[str] = []
        for date_index, trade_date in enumerate(trade_dates, start=1):
            refresh = force_refresh or trade_date in automatically_refreshed
            daily = self._download_endpoint_for_date(
                "daily",
                self.DAILY_FIELDS,
                trade_date,
                refresh,
                stats,
            )
            if daily.empty:
                warning = (
                    f"{trade_date} 的 daily 尚无数据，已跳过；"
                    "若为当日数据，请在收盘入库后重试"
                )
                stats.warnings.append(warning)
                self.progress(warning)
                continue

            daily_basic = self._download_endpoint_for_date(
                "daily_basic",
                self.DAILY_BASIC_FIELDS,
                trade_date,
                refresh,
                stats,
            )
            adj_factor = self._download_endpoint_for_date(
                "adj_factor",
                self.ADJ_FACTOR_FIELDS,
                trade_date,
                refresh,
                stats,
            )
            if daily_basic.empty or adj_factor.empty:
                raise RuntimeError(
                    f"{trade_date} is missing daily_basic or adj_factor data"
                )
            usable_dates.append(trade_date)
            if date_index == 1 or date_index % 20 == 0 or date_index == len(trade_dates):
                self.progress(
                    f"Tushare 数据进度 {date_index}/{len(trade_dates)}，"
                    f"当前日期 {trade_date}"
                )

        if not usable_dates:
            raise RuntimeError("No usable Tushare daily data was downloaded")

        stock_basic = self.fetch_stock_basic(force_refresh=refresh_master)
        name_history = self.fetch_name_history(
            force_refresh=refresh_master, warnings=stats.warnings
        )
        market = self.build_standard_market_data(
            usable_dates, stock_basic, name_history
        )
        stats.output_rows = len(market)
        stats.latest_trade_date = (
            pd.Timestamp(market["date"].max()).date().isoformat()
        )
        return market, stats

    def _load_partitions(self, endpoint: str, trade_dates: list[str]) -> pd.DataFrame:
        frames = [
            self._read_csv(self._cache_path(endpoint, trade_date))
            for trade_date in trade_dates
            if self._cache_path(endpoint, trade_date).exists()
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _require_columns(
        frame: pd.DataFrame, columns: list[str], endpoint: str
    ) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{endpoint} data is missing fields: {missing}")

    def build_standard_market_data(
        self,
        trade_dates: list[str],
        stock_basic: pd.DataFrame,
        name_history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        daily = self._load_partitions("daily", trade_dates)
        daily_basic = self._load_partitions("daily_basic", trade_dates)
        adj_factor = self._load_partitions("adj_factor", trade_dates)
        self._require_columns(
            daily,
            [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "vol",
                "amount",
            ],
            "daily",
        )
        self._require_columns(
            daily_basic,
            ["ts_code", "trade_date", "turnover_rate", "total_mv"],
            "daily_basic",
        )
        self._require_columns(
            adj_factor,
            ["ts_code", "trade_date", "adj_factor"],
            "adj_factor",
        )

        basic_columns = [
            column
            for column in [
                "ts_code",
                "trade_date",
                "turnover_rate",
                "total_mv",
                "circ_mv",
                "limit_status",
            ]
            if column in daily_basic.columns
        ]
        market = daily.merge(
            daily_basic[basic_columns],
            on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        ).merge(
            adj_factor[["ts_code", "trade_date", "adj_factor"]],
            on=["ts_code", "trade_date"],
            how="left",
            validate="one_to_one",
        )

        stock_columns = [
            column
            for column in [
                "ts_code",
                "symbol",
                "name",
                "industry",
                "market",
                "exchange",
                "list_status",
                "list_date",
                "delist_date",
            ]
            if column in stock_basic.columns
        ]
        market = market.merge(
            stock_basic[stock_columns].drop_duplicates("ts_code"),
            on="ts_code",
            how="left",
            validate="many_to_one",
        )

        market["date"] = pd.to_datetime(market["trade_date"], errors="coerce")
        market["code"] = (
            market.get("symbol", market["ts_code"].str.split(".").str[0])
            .astype("string")
            .str.zfill(6)
        )
        market["industry"] = (
            market.get("industry", pd.Series(index=market.index, dtype="string"))
            .astype("string")
            .fillna("UNKNOWN")
            .replace("", "UNKNOWN")
        )

        for column in ["open", "high", "low", "close", "vol", "amount"]:
            market[column] = pd.to_numeric(market[column], errors="coerce")
        market["volume"] = market["vol"] * 100.0
        market["amount"] = market["amount"] * 1_000.0
        market["adj_factor"] = pd.to_numeric(
            market["adj_factor"], errors="coerce"
        ).fillna(1.0)
        market["market_cap"] = (
            pd.to_numeric(market["total_mv"], errors="coerce") * 10_000.0
        )
        market["turnover_rate"] = (
            pd.to_numeric(market["turnover_rate"], errors="coerce") / 100.0
        )
        list_date = pd.to_datetime(market.get("list_date"), errors="coerce")
        market["listing_days"] = (market["date"] - list_date).dt.days + 1

        limit_status = pd.to_numeric(
            market.get(
                "limit_status", pd.Series(0, index=market.index, dtype=float)
            ),
            errors="coerce",
        ).fillna(0)
        market["is_limit_up"] = limit_status.isin([2, 3])
        market["is_limit_down"] = limit_status.isin([5, 6])
        market["is_suspended"] = False
        market["is_st"] = self._historical_st_flags(
            market, stock_basic, name_history
        )
        market["has_valid_market_data"] = (
            market[["open", "high", "low", "close"]].gt(0).all(axis=1)
            & market[["volume", "amount"]].ge(0).all(axis=1)
        )
        for price_column in ["open", "high", "low", "close"]:
            market[f"adj_{price_column}"] = (
                market[price_column] * market["adj_factor"]
            )

        output_columns = [
            "date",
            "code",
            "ts_code",
            "name",
            "industry",
            "market",
            "exchange",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "adj_factor",
            "market_cap",
            "turnover_rate",
            "listing_days",
            "is_st",
            "is_suspended",
            "is_limit_up",
            "is_limit_down",
            "has_valid_market_data",
            "adj_open",
            "adj_high",
            "adj_low",
            "adj_close",
        ]
        market = market[output_columns].dropna(
            subset=["date", "code", "open", "high", "low", "close"]
        )
        return market.sort_values(["date", "code"], kind="stable").reset_index(
            drop=True
        )

    @staticmethod
    def _historical_st_flags(
        market: pd.DataFrame,
        stock_basic: pd.DataFrame,
        name_history: pd.DataFrame | None,
    ) -> pd.Series:
        current_names = (
            stock_basic.set_index("ts_code")["name"].astype("string")
            if "name" in stock_basic.columns
            else pd.Series(dtype="string")
        )
        current_is_st = market["ts_code"].map(
            current_names.str.upper().str.contains("ST", na=False)
        ).fillna(False)
        if name_history is None or name_history.empty:
            return current_is_st.astype(bool)

        required = {"ts_code", "name", "start_date", "end_date"}
        if not required.issubset(name_history.columns):
            return current_is_st.astype(bool)

        result = pd.Series(False, index=market.index, dtype=bool)
        history = name_history.copy()
        history["is_st_name"] = (
            history["name"].astype("string").str.upper().str.contains("ST", na=False)
        )
        history = history.loc[history["is_st_name"]]
        codes_with_history = set(name_history["ts_code"].dropna().astype(str))

        for ts_code, row_indices in market.groupby("ts_code").groups.items():
            indices = np.asarray(list(row_indices), dtype=int)
            if str(ts_code) not in codes_with_history:
                result.iloc[indices] = current_is_st.iloc[indices].to_numpy()
                continue
            dates = market.loc[indices, "date"].to_numpy(dtype="datetime64[D]")
            flags = np.zeros(len(indices), dtype=bool)
            intervals = history.loc[history["ts_code"].astype(str).eq(str(ts_code))]
            for interval in intervals.itertuples(index=False):
                start = pd.to_datetime(interval.start_date, errors="coerce")
                end = pd.to_datetime(interval.end_date, errors="coerce")
                if pd.isna(start):
                    continue
                start_value = np.datetime64(start.date())
                end_value = (
                    np.datetime64(end.date())
                    if not pd.isna(end)
                    else np.datetime64("2262-04-11")
                )
                flags |= (dates >= start_value) & (dates <= end_value)
            result.iloc[indices] = flags
        return result
