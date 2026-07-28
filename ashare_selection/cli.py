from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import load_config, write_default_config
from .demo import generate_demo_data
from .pipeline import (
    CandidateSelector,
    write_backtest_result,
    write_selection_result,
)
from .tushare_source import TushareDataSource


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ashare_selection",
        description="A股横截面候选池研究工具",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser(
        "init-config", help="生成一份默认 JSON 配置"
    )
    config_parser.add_argument("--output", default="config.json")

    demo_parser = subparsers.add_parser(
        "make-demo-data", help="生成仅用于功能验证的合成日线数据"
    )
    demo_parser.add_argument("--output", default="data/demo_market.csv")
    demo_parser.add_argument("--stocks", type=int, default=80)
    demo_parser.add_argument("--days", type=int, default=520)
    demo_parser.add_argument("--seed", type=int, default=7)

    select_parser = subparsers.add_parser(
        "select", help="对指定日期生成 Top-K 候选池"
    )
    select_parser.add_argument("--input", required=True)
    select_parser.add_argument("--config", default="config.json")
    select_parser.add_argument("--output", default="output/latest")
    select_parser.add_argument("--as-of")
    select_parser.add_argument("--previous")

    backtest_parser = subparsers.add_parser(
        "backtest", help="运行严格时间滚动的候选池回测"
    )
    backtest_parser.add_argument("--input", required=True)
    backtest_parser.add_argument("--config", default="config.json")
    backtest_parser.add_argument("--output", default="output/backtest")

    tushare_download_parser = subparsers.add_parser(
        "tushare-download", help="从 Tushare 增量下载并转换 A 股日线数据"
    )
    tushare_download_parser.add_argument("--start-date", required=True)
    tushare_download_parser.add_argument(
        "--end-date", default=pd.Timestamp.today().strftime("%Y%m%d")
    )
    tushare_download_parser.add_argument("--config", default="config.json")
    tushare_download_parser.add_argument(
        "--output", default="data/tushare_market.csv"
    )
    tushare_download_parser.add_argument("--cache-dir")
    tushare_download_parser.add_argument("--force-refresh", action="store_true")
    tushare_download_parser.add_argument("--refresh-master", action="store_true")

    tushare_select_parser = subparsers.add_parser(
        "tushare-select", help="下载 Tushare 数据并直接生成实际候选池"
    )
    tushare_select_parser.add_argument("--start-date", required=True)
    tushare_select_parser.add_argument(
        "--end-date", default=pd.Timestamp.today().strftime("%Y%m%d")
    )
    tushare_select_parser.add_argument("--config", default="config.json")
    tushare_select_parser.add_argument("--output", default="output/tushare_latest")
    tushare_select_parser.add_argument("--market-output")
    tushare_select_parser.add_argument("--cache-dir")
    tushare_select_parser.add_argument("--as-of")
    tushare_select_parser.add_argument("--previous")
    tushare_select_parser.add_argument("--force-refresh", action="store_true")
    tushare_select_parser.add_argument("--refresh-master", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "init-config":
        output = write_default_config(args.output)
        print(json.dumps({"config": str(output)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "make-demo-data":
        output = generate_demo_data(
            args.output, stocks=args.stocks, days=args.days, seed=args.seed
        )
        print(json.dumps({"demo_data": str(output)}, ensure_ascii=False, indent=2))
        return 0

    config = load_config(args.config)
    if getattr(args, "cache_dir", None):
        config.tushare.cache_dir = args.cache_dir

    if args.command in {"tushare-download", "tushare-select"}:
        source = TushareDataSource(
            config,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        market, stats = source.download(
            args.start_date,
            args.end_date,
            force_refresh=args.force_refresh,
            refresh_master=args.refresh_master,
        )

        if args.command == "tushare-download":
            market_path = Path(args.output)
            market_path.parent.mkdir(parents=True, exist_ok=True)
            market.to_csv(market_path, index=False)
            print(
                json.dumps(
                    {
                        "download": asdict(stats),
                        "market_data": str(market_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        if args.market_output:
            market_path = Path(args.market_output)
            market_path.parent.mkdir(parents=True, exist_ok=True)
            market.to_csv(market_path, index=False)
        previous_codes = None
        if args.previous:
            previous = pd.read_csv(args.previous, dtype={"code": "string"})
            if "code" not in previous.columns:
                raise ValueError("Previous candidate file must contain a 'code' column")
            previous_codes = previous["code"].dropna().astype(str).tolist()
        selector = CandidateSelector(config)
        prepared = selector.prepare(market)
        result = selector.select_prepared(
            prepared, as_of=args.as_of, previous_codes=previous_codes
        )
        paths = write_selection_result(result, args.output)
        print(
            json.dumps(
                {
                    "download": asdict(stats),
                    "diagnostics": asdict(result.diagnostics),
                    "market_data": args.market_output,
                    "files": {name: str(path) for name, path in paths.items()},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    selector = CandidateSelector(config)

    if args.command == "select":
        result = selector.select_from_csv(
            args.input,
            as_of=args.as_of,
            previous_candidates_path=args.previous,
        )
        paths = write_selection_result(result, args.output)
        payload = {
            "diagnostics": asdict(result.diagnostics),
            "files": {name: str(path) for name, path in paths.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "backtest":
        prepared = selector.prepare_from_csv(args.input)
        result = selector.backtest_prepared(prepared)
        paths = write_backtest_result(result, args.output)
        payload = {
            "summary": result.summary,
            "files": {name: str(path) for name, path in paths.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    raise RuntimeError(f"Unhandled command: {args.command}")
