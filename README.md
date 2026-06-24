# 商品期货交易系统

基于 Dash + Plotly 的商品期货量化分析与可视化系统。拉取国内期货主力合约日线数据，
完成品种评分排名、市场状态分析、信号生成、风险管理与历史回测，并以交互式 Web 仪表板呈现。

## 功能模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据获取 | `data_fetcher.py` | akshare 拉取主力合约日线，失败时自动生成仿真数据 |
| 技术指标 | `indicators.py` | MA / ATR / 布林带 / 等指标计算 |
| 品种选择 | `variety_selector.py` | 多因子评分对品种排名，输出多空方向 |
| 市场分析 | `market_analyzer.py` | 各品种市场状态、入场/止损/止盈位 |
| 信号生成 | `signal_generator.py` | 交易信号 |
| 风险管理 | `risk_manager.py` | 仓位与风险控制 |
| 回测 | `backtester.py` | 组合历史回测，输出收益/胜率/夏普 |
| 可视化 | `dashboard.py` | Dash 交互式仪表板 |
| 主入口 | `main.py` | 串联全流程并启动仪表板 |

## 运行

```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install pyarrow      # parquet 缓存引擎

# 2. 启动
python main.py

# 3. 浏览器访问
http://127.0.0.1:8050
```

## 数据说明

- 优先通过 **akshare** 获取真实行情（`futures_zh_daily_sina`）。
- 当网络不可用或数据不足 60 条时，自动回退到**仿真数据**（带趋势 + 周期 + 噪声），
  保证系统在离线环境下也能完整跑通演示。
- 数据缓存为 parquet 格式，缓存有效期见 `config.py` 中 `DATA_PARAMS["cache_hours"]`。

## 依赖

akshare · pandas · numpy · plotly · dash · dash-bootstrap-components · pyarrow

## 免责声明

本系统仅用于技术研究与学习演示，不构成任何投资建议。期货交易风险极高，据此操作风险自负。
