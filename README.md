# A 股动态候选池 POC

这是一个面向研究和回测的第一版实现：先对全部可交易 A 股进行低成本横截面扫描，再输出较小的 Top-K 候选池，供后续 AI 阅读公告、财报、新闻并进行更深入的买卖判断。

当前版本聚焦价格成交量数据，借鉴《Generative AI for Stock Selection》的三个核心思想：

- 所有特征必须只使用当时已经可见的数据；
- 优先使用横截面排名、波动率归一化、多周期比较和非线性交互；
- 把传统特征和 AI 后续生成的特征交给统一、可验证的滚动训练管线。

## 当前具备的能力

- CSV 数据校验、股票代码和布尔字段规范化；
- ST、停牌、涨停、上市天数、价格和成交额过滤；
- 动量、反转、波动率、隔夜跳空、成交量、价格区间、Amihud 非流动性等特征；
- 每日横截面百分位排名；
- 使用下一交易日开盘到未来第 H 个交易日开盘的收益作为标签；
- 严格按时间截断训练标签，避免未来数据泄漏；
- 无外部机器学习依赖的岭回归排序基线；
- 波动率和非流动性惩罚；
- Top-K、行业上限和候选池进出缓冲；
- 滚动回测、Precision@K、IC、超额收益和换手率输出。

> 本版本的岭回归是用于打通数据和验证流程的基线。正式研究应继续加入 LightGBM Ranker，并与当前基线做严格样本外比较。

## 安装

需要 Python 3.11 或以上：

```bash
python3 -m pip install -r requirements.txt
```

## 使用 Tushare 做实际筛选

完整流程需要至少能调用 `trade_cal`、`stock_basic`、`daily`、`daily_basic` 和 `adj_factor`。根据 Tushare 当前权限说明，其中多项接口要求至少 2000 积分：

- [A 股日线 daily](https://tushare.pro/document/2?doc_id=27)
- [每日指标 daily_basic](https://tushare.pro/document/2?doc_id=32)
- [复权因子 adj_factor](https://tushare.pro/document/2?doc_id=28)
- [股票基础信息 stock_basic](https://tushare.pro/document/2?doc_id=25)
- [交易日历 trade_cal](https://tushare.pro/document/2?doc_id=26)

### 1. 安全设置 Token

Token 只通过环境变量读取，不要写进 Git、Python 文件或 JSON 配置：

```bash
export TUSHARE_TOKEN='你的真实Token'
```

可以检查环境变量是否存在，但不要直接打印完整 Token：

```bash
python3 -c "import os; print('TUSHARE_TOKEN 已设置' if os.getenv('TUSHARE_TOKEN') else '未设置')"
```

### 2. 下载并直接生成候选池

当前模型使用 500 个交易日训练，并需要额外的特征预热和标签间隔。首次运行建议准备至少约 650 个交易日的数据。例如：

```bash
python3 -m ashare_selection tushare-select \
  --start-date 20240101 \
  --config config.tushare.json \
  --output output/tushare_latest
```

`--end-date` 默认是执行当天。若当天数据尚未入库，程序会跳过当天并使用最近一个已经取得数据的交易日。

候选结果：

```text
output/tushare_latest/candidates.csv
output/tushare_latest/scored_universe.csv
output/tushare_latest/selection_diagnostics.json
```

如需同时保存转换后的统一 CSV：

```bash
python3 -m ashare_selection tushare-select \
  --start-date 20240101 \
  --config config.tushare.json \
  --market-output data/tushare_market.csv \
  --output output/tushare_latest
```

### 3. 增量更新

API 原始数据按接口和交易日缓存到：

```text
data/tushare_cache/daily/
data/tushare_cache/daily_basic/
data/tushare_cache/adj_factor/
data/tushare_cache/master/
```

以后重复执行同一条 `tushare-select` 命令，历史日期直接读取本地缓存，只重新获取配置中指定的最近两个交易日。

有新股上市、退市或行业基础信息需要更新时：

```bash
python3 -m ashare_selection tushare-select \
  --start-date 20240101 \
  --config config.tushare.json \
  --refresh-master \
  --output output/tushare_latest
```

### 4. 只下载，不运行筛选

```bash
python3 -m ashare_selection tushare-download \
  --start-date 20240101 \
  --config config.tushare.json \
  --output data/tushare_market.csv
```

### Tushare 单位与过滤处理

程序会自动进行以下转换：

- `daily.vol`：手 → 股，乘以 100；
- `daily.amount`：千元 → 元，乘以 1000；
- `daily_basic.total_mv`：万元 → 元，乘以 10000；
- `daily_basic.turnover_rate`：百分数 → 小数，除以 100；
- 使用 `adj_factor` 构造复权价格；
- 使用 `limit_status` 识别涨停和跌停；
- 使用 `namechange` 的历史名称区间识别当时的 ST 状态；
- 停牌股票在 `daily` 中没有记录，因此不会进入当日候选池；
- 如果未来入场日或退出日停牌，该训练标签会设为缺失，不会把下一次复牌日错误当成指定交易日。

## 先运行合成数据演示

合成数据仅用于验证程序能否运行，不代表真实收益。

```bash
python3 -m ashare_selection make-demo-data \
  --output data/demo_market.csv \
  --stocks 80 \
  --days 520

python3 -m ashare_selection select \
  --input data/demo_market.csv \
  --config config.demo.json \
  --output output/demo_select

python3 -m ashare_selection backtest \
  --input data/demo_market.csv \
  --config config.demo.json \
  --output output/demo_backtest
```

## 接入真实 A 股数据

复制并调整配置：

```bash
cp config.example.json config.json
```

输入 CSV 每行代表一只股票的一个交易日。必需字段：

| 字段 | 含义 |
|---|---|
| `date` | 交易日期 |
| `code` | 六位股票代码 |
| `open/high/low/close` | 当日未复权 OHLC |
| `volume` | 成交股数 |
| `amount` | 成交金额，单位必须在全数据中一致 |

强烈建议提供：

| 字段 | 含义 |
|---|---|
| `industry` | 当时有效的行业分类 |
| `adj_factor` | 当时可用的复权因子 |
| `market_cap` | 当日总市值或流通市值，口径保持一致 |
| `turnover_rate` | 当日换手率 |
| `listing_days` | 截至当日的上市交易天数 |
| `is_st` | 当日是否风险警示 |
| `is_suspended` | 当日是否停牌 |
| `is_limit_up` | 当日收盘时是否涨停、下一时点难以买入 |
| `is_limit_down` | 当日是否跌停 |

如果缺少可选布尔字段，程序会按 `False` 处理；这只适合功能测试，不适合真实回测。行业、ST、停牌、涨跌停和历史复权状态都应使用 point-in-time 数据，不能用今天的状态回填历史。

生成最新候选池：

```bash
python3 -m ashare_selection select \
  --input data/ashare_daily.csv \
  --config config.json \
  --output output/latest
```

指定历史评分日期：

```bash
python3 -m ashare_selection select \
  --input data/ashare_daily.csv \
  --config config.json \
  --as-of 2026-06-30 \
  --output output/2026-06-30
```

使用上一期候选池启用名单缓冲：

```bash
python3 -m ashare_selection select \
  --input data/ashare_daily.csv \
  --config config.json \
  --previous output/previous/candidates.csv \
  --output output/latest
```

## 输出说明

选股命令生成：

- `candidates.csv`：供后续 AI 使用的 Top-K 候选池；
- `scored_universe.csv`：全部合格股票的模型分数和排名；
- `selection_diagnostics.json`：训练区间、样本量和过滤结果。

回测命令生成：

- `backtest_periods.csv`：每个调仓期的候选池表现；
- `backtest_holdings.csv`：历史每期入选股票；
- `backtest_summary.json`：IC、Precision@K、收益和换手率汇总。

## 关键设计

信号在交易日 `t` 收盘后形成。标签定义为：

```text
return = adjusted_open[t + H + 1] / adjusted_open[t + 1] - 1
```

即假设最早在下一交易日开盘成交，并持有 H 个交易日。训练评分日的模型时，最后一个训练标签的退出价格不得晚于评分日，从代码层面隔离未来信息。

最终候选分数为：

```text
候选分数
= 模型预测的横截面百分位
- 波动率惩罚
- 非流动性惩罚
```

随后执行行业数量上限和候选池缓冲。真实项目建议同时比较 `K = 50、100、200、300、500`，选择能保留主要信号且满足后续 AI 成本限制的最小 K。

## 尚未包含

- 手续费、卖出税费、冲击成本和涨跌停排队的完整成交模拟；
- 财报实际披露时间、分析师预测和公告文本；
- LightGBM LambdaRank、模型集成和超参数时间滚动验证；
- 多重检验校正和 LLM 自动生成特征的代码沙箱；
- 实盘下单、仓位管理或任何收益保证。

在以上项目完成并通过独立样本外验证前，本项目只应作为研究工具。

## 测试

```bash
python3 -m unittest discover -s tests -v
```


