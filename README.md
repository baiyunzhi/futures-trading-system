# 商品期货交易系统

基于 Dash + Plotly 的商品期货量化分析与可视化系统。拉取国内期货主力合约日线数据，
完成品种评分排名、市场状态分析、信号生成、风险管理与历史回测，并以交互式 Web 仪表板呈现。

## 功能模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据获取 | `data_fetcher.py` | akshare 拉取主力合约日线，失败时自动生成仿真数据 |
| 持仓量数据 | `open_interest_sources.py` | 总持仓量可用性检查、会员持仓排名查询 |
| 技术指标 | `indicators.py` | MA / ATR / 布林带 / 等指标计算 |
| 品种选择 | `variety_selector.py` | 多因子评分对品种排名，输出多空方向 |
| 市场分析 | `market_analyzer.py` | 各品种市场状态、入场/止损/止盈位 |
| 四维行情引擎 | `objective_market_engine.py` | 价格、成交量、持仓量、时间的客观描述与下注计划 |
| 信号生成 | `signal_generator.py` | 交易信号 |
| 风险管理 | `risk_manager.py` | 仓位与风险控制 |
| 回测 | `backtester.py` | 组合历史回测，输出收益/胜率/夏普 |
| 本地模拟盘 | `paper_trading.py` / `run_paper.py` | 回放撮合、委托、成交、持仓、资金曲线 |
| 可视化 | `dashboard.py` | Dash 交互式仪表板 |
| 主入口 | `main.py` | 串联全流程并启动仪表板 |

## 运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动
python main.py

# 3. 浏览器访问
http://127.0.0.1:8050
```

## 当前核心流程

系统先描述行情，再决定是否下注：

```text
周线大环境
+ 日线主状态
+ 小时线触发条件
+ 价格结构
+ 成交量主动性
+ 持仓量持续性
+ 风险手数
= 客观行情描述与下注计划
```

仪表板新增“四维客观行情描述与下注计划”表，直接输出：

- 当下行情正在干什么
- 方向是否明确
- 持续性评分
- 是否允许下注
- 入场触发条件
- 止损、1R、2R、建议手数

详细规则见 `docs/objective_market_system.md`。

## 持仓量数据

已修复 akshare 日线字段映射：

```text
futures_zh_daily_sina 返回 hold
系统映射为 open_interest
四维行情引擎用 open_interest 判断持续性
```

可检查当前数据是否有持仓量：

```bash
python open_interest_sources.py
```

会员多空持仓排名需要真实合约代码：

```python
from open_interest_sources import fetch_sina_member_position_rank

rank = fetch_sina_member_position_rank("RB2410", "2024-06-21")
```

`RB0` 这类主力连续代码适合日线行情，不适合会员持仓排名；排名数据要用 `RB2410`、`I2409` 这类具体合约。

## 生成静态网页版

```bash
python export_static.py
```

生成文件：`docs/index.html`

GitHub Pages 可直接选择 `main` 分支的 `/docs` 目录发布。

## 本地模拟盘

```bash
python run_paper.py
```

输出目录：`paper_trading/`

| 文件 | 内容 |
|------|------|
| `account.json` | 模拟账户摘要 |
| `orders.csv` | 委托记录 |
| `fills.csv` | 成交记录 |
| `positions.csv` | 当前持仓 |
| `equity_curve.csv` | 权益曲线 |

模拟盘只做本地历史回放撮合，不连接真实期货账户，不会真实下单。

## 数据说明

- 优先通过 **akshare** 获取真实行情（`futures_zh_daily_sina`）。
- 当网络不可用或数据不足 60 条时，自动回退到**仿真数据**（带趋势 + 周期 + 噪声），
  保证系统在离线环境下也能完整跑通演示。
- 数据缓存为 CSV 格式，缓存有效期见 `config.py` 中 `DATA_PARAMS["cache_hours"]`。

## 已修复的关键漏洞

- 回测成交从“信号日收盘价成交”改为“下一根 K 线开盘价成交”，避免不可成交价格和前视偏差。
- 回测加入滑点、合约乘数、ATR 风险手数，避免固定 1 手导致风险失真。
- 止损止盈改为检查日内 high/low，避免只看收盘价漏掉硬止损。
- 交易 PnL 计入开平双边手续费，净值曲线加入持仓浮盈浮亏。
- 最后一根 K 线仍有持仓时强制平仓，避免绩效遗漏未平仓风险。
- 静态页面导出不再强依赖 Dash，可用于 GitHub Pages。

## 剩余风险

- akshare 数据接口和主力连续合约规则可能变化，实盘前必须校验复权、换月和缺失数据。
- 当前策略基于日线，不适合作为自动下单系统直接实盘。
- 若同一日同时触发止损和止盈，回测按保守顺序优先止损。
- 仿真数据仅用于离线演示，不能用于策略有效性结论。

详见 `docs/AUDIT.md`。

## 依赖

akshare · pandas · numpy · plotly · dash · dash-bootstrap-components · requests

## 免责声明

本系统仅用于技术研究与学习演示，不构成任何投资建议。期货交易风险极高，据此操作风险自负。
