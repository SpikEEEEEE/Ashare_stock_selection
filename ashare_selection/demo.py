from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(
    output_path: str | Path,
    stocks: int = 80,
    days: int = 520,
    seed: int = 7,
) -> Path:
    if stocks < 20:
        raise ValueError("Demo generation needs at least 20 stocks")
    if days < 300:
        raise ValueError("Demo generation needs at least 300 trading days")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=days)
    codes = [f"{index + 1:06d}" for index in range(stocks)]
    industry_names = [
        "银行",
        "电子",
        "医药",
        "机械",
        "消费",
        "化工",
        "公用事业",
        "计算机",
        "汽车",
        "建筑",
    ]
    industry_id = np.arange(stocks) % len(industry_names)

    daily_returns = np.zeros((days, stocks), dtype=float)
    market_shocks = rng.normal(0.00015, 0.007, size=days)
    industry_shocks = rng.normal(
        0.0, 0.004, size=(days, len(industry_names))
    )
    quality = rng.normal(0.0, 1.0, size=stocks)

    for day in range(days):
        if day >= 5:
            momentum = daily_returns[day - 5 : day].mean(axis=0)
        else:
            momentum = np.zeros(stocks)
        previous = daily_returns[day - 1] if day else np.zeros(stocks)
        idiosyncratic = rng.normal(0.0, 0.014, size=stocks)
        daily_returns[day] = np.clip(
            market_shocks[day]
            + industry_shocks[day, industry_id]
            + 0.12 * momentum
            - 0.07 * previous
            + 0.00015 * quality
            + idiosyncratic,
            -0.095,
            0.095,
        )

    previous_close = rng.uniform(6.0, 60.0, size=stocks)
    shares_outstanding = rng.uniform(80_000_000, 2_000_000_000, size=stocks)
    base_volume = rng.uniform(2_000_000, 30_000_000, size=stocks)
    rows: list[dict[str, object]] = []

    for day_index, date in enumerate(dates):
        overnight = rng.normal(0.0, 0.0035, size=stocks)
        intraday = daily_returns[day_index] - overnight
        open_price = previous_close * (1.0 + overnight)
        close_price = open_price * (1.0 + intraday)
        spread = np.abs(rng.normal(0.009, 0.004, size=stocks))
        high = np.maximum(open_price, close_price) * (1.0 + spread)
        low = np.minimum(open_price, close_price) * (1.0 - spread)
        volume_multiplier = np.exp(
            rng.normal(0.0, 0.45, size=stocks)
            + 5.0 * np.abs(daily_returns[day_index])
        )
        volume = base_volume * volume_multiplier

        suspended = rng.random(stocks) < 0.002
        open_price[suspended] = previous_close[suspended]
        close_price[suspended] = previous_close[suspended]
        high[suspended] = previous_close[suspended]
        low[suspended] = previous_close[suspended]
        volume[suspended] = 0.0
        amount = volume * (open_price + close_price) / 2.0

        limit_up = daily_returns[day_index] >= 0.094
        limit_down = daily_returns[day_index] <= -0.094
        st_mask = (
            (np.arange(stocks) < max(1, stocks // 40))
            & (day_index > int(days * 0.75))
        )

        for stock_index, code in enumerate(codes):
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "code": code,
                    "industry": industry_names[industry_id[stock_index]],
                    "open": round(float(open_price[stock_index]), 4),
                    "high": round(float(high[stock_index]), 4),
                    "low": round(float(low[stock_index]), 4),
                    "close": round(float(close_price[stock_index]), 4),
                    "volume": round(float(volume[stock_index]), 2),
                    "amount": round(float(amount[stock_index]), 2),
                    "adj_factor": 1.0,
                    "market_cap": round(
                        float(close_price[stock_index] * shares_outstanding[stock_index]),
                        2,
                    ),
                    "turnover_rate": float(
                        volume[stock_index] / shares_outstanding[stock_index]
                    ),
                    "listing_days": 300 + day_index,
                    "is_st": bool(st_mask[stock_index]),
                    "is_suspended": bool(suspended[stock_index]),
                    "is_limit_up": bool(limit_up[stock_index]),
                    "is_limit_down": bool(limit_down[stock_index]),
                }
            )
        previous_close = close_price

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output

