from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import load_config, write_default_config
from .data import load_market_data
from .deepseek_features import (
    DeepSeekFeatureGenerator,
    write_deepseek_generation_result,
)
from .demo import generate_demo_data
from .generated_features import (
    load_feature_definitions,
    screen_feature_definitions,
    write_feature_screening_result,
)
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
    select_parser.add_argument("--generated-features")

    backtest_parser = subparsers.add_parser(
        "backtest", help="运行严格时间滚动的候选池回测"
    )
    backtest_parser.add_argument("--input", required=True)
    backtest_parser.add_argument("--config", default="config.json")
    backtest_parser.add_argument("--output", default="output/backtest")
    backtest_parser.add_argument("--generated-features")
    backtest_parser.add_argument("--start-date")
    backtest_parser.add_argument("--end-date")

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
    tushare_select_parser.add_argument("--generated-features")
    tushare_select_parser.add_argument("--force-refresh", action="store_true")
    tushare_select_parser.add_argument("--refresh-master", action="store_true")

    deepseek_parser = subparsers.add_parser(
        "deepseek-generate-features",
        help="调用 DeepSeek 生成受限 DSL 特征候选",
    )
    deepseek_parser.add_argument("--config", default="config.json")
    deepseek_parser.add_argument(
        "--output",
        default="data/deepseek_features/proposals.json",
    )
    deepseek_parser.add_argument("--count", type=int)

    screen_parser = subparsers.add_parser(
        "screen-features",
        help="按覆盖率、日度 IC 和相关性筛选生成特征",
    )
    screen_parser.add_argument("--input", required=True)
    screen_parser.add_argument("--definitions", required=True)
    screen_parser.add_argument("--config", default="config.json")
    screen_parser.add_argument(
        "--output",
        default="data/deepseek_features/screened",
    )
    screen_parser.add_argument("--start-date")
    screen_parser.add_argument("--end-date")
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
    if getattr(args, "generated_features", None):
        config.features.generated_feature_path = args.generated_features

    if args.command == "deepseek-generate-features":
        generator = DeepSeekFeatureGenerator(config)
        result = generator.generate(args.count)
        output = write_deepseek_generation_result(result, args.output)
        print(
            json.dumps(
                {
                    "model": result.model,
                    "valid_features": len(result.features),
                    "rejected_features": len(result.rejected),
                    "usage": result.usage,
                    "output": str(output),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.command == "screen-features":
        definitions = load_feature_definitions(args.definitions)
        market = load_market_data(args.input, config)
        result = screen_feature_definitions(
            market,
            config,
            definitions,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        paths = write_feature_screening_result(result, args.output)
        print(
            json.dumps(
                {
                    "screening_start": result.screening_start,
                    "screening_end": result.screening_end,
                    "label_data_end": result.label_data_end,
                    "screening_rows": result.screening_rows,
                    "candidate_features": result.candidate_features,
                    "accepted_features": len(result.accepted),
                    "files": {name: str(path) for name, path in paths.items()},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

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
        result = selector.backtest_prepared(
            prepared,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        paths = write_backtest_result(result, args.output)
        payload = {
            "summary": result.summary,
            "files": {name: str(path) for name, path in paths.items()},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    raise RuntimeError(f"Unhandled command: {args.command}")
