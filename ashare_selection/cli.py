from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import load_config, write_default_config
from .demo import generate_demo_data
from .pipeline import (
    CandidateSelector,
    write_backtest_result,
    write_selection_result,
)


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

