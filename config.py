# ============================================================
#  交易系统配置文件
#  标的：中国商品期货（黑色系 / 有色金属 / 农产品）
# ============================================================

# ---------- 品种池 ----------
VARIETIES = {
    "黑色系": {
        "RB0": "螺纹钢",
        "HC0": "热卷",
        "I0":  "铁矿石",
        "J0":  "焦炭",
        "JM0": "焦煤",
    },
    "有色金属": {
        "CU0": "铜",
        "AL0": "铝",
        "ZN0": "锌",
        "NI0": "镍",
    },
    "农产品": {
        "M0":  "豆粕",
        "Y0":  "豆油",
        "C0":  "玉米",
        "SR0": "白糖",
        "CF0": "棉花",
    },
}

# 所有品种扁平化映射  symbol → 中文名
ALL_SYMBOLS: dict[str, str] = {}
SYMBOL_SECTOR: dict[str, str] = {}
for _sector, _items in VARIETIES.items():
    for _sym, _name in _items.items():
        ALL_SYMBOLS[_sym] = _name
        SYMBOL_SECTOR[_sym] = _sector

# ---------- 必要指标参数 ----------
INDICATOR_PARAMS = {
    "ATR":  14,                  # ATR 周期
}

# ---------- 品种评分权重（合计=1.0）----------
SCORE_WEIGHTS = {
    "trend":      0.40,   # 趋势强度
    "momentum":   0.35,   # 近期动量
    "volatility": 0.25,   # 适度波动
}

# ---------- 市场状态阈值 ----------
# 综合评分 0-100
STATE_THRESHOLDS = {
    "trend_long":   65,   # 分数 >= 65 且 方向向上 → 趋势做多
    "trend_short":  65,   # 分数 >= 65 且 方向向下 → 趋势做空
    "light_trade":  45,   # 45 <= 分数 < 65 → 轻仓短线
    "observe":       0,   # 分数 < 45 → 观望
}

# ---------- 风险管理参数 ----------
RISK_PARAMS = {
    "capital":            500_000,   # 初始资金（元）
    "max_risk_per_trade": 0.02,      # 单笔最大亏损占总资金比例
    "max_positions":      3,         # 最多同时持仓品种数
    "atr_stop_mult":      2.0,       # 止损 = 入场价 ± N×ATR
    "atr_target_mult":    3.0,       # 止盈 = 入场价 ± N×ATR
    "commission_rate":    0.0001,    # 手续费率（单边万分之一）
    "slippage_ticks":     1,         # 滑点（跳数）
    "default_tick_size":   1.0,       # 默认最小变动价位
    "default_lot_value":   1.0,       # 默认合约乘数（元/点/手）
}

# ---------- 合约参数 ----------
# 仅用于回测风控估算；缺失品种回落到 RISK_PARAMS 默认值。
CONTRACT_SPECS = {
    "RB0": {"lot_value": 10, "tick_size": 1},
    "HC0": {"lot_value": 10, "tick_size": 1},
    "I0":  {"lot_value": 100, "tick_size": 0.5},
    "J0":  {"lot_value": 100, "tick_size": 0.5},
    "JM0": {"lot_value": 60, "tick_size": 0.5},
    "CU0": {"lot_value": 5, "tick_size": 10},
    "AL0": {"lot_value": 5, "tick_size": 5},
    "ZN0": {"lot_value": 5, "tick_size": 5},
    "NI0": {"lot_value": 1, "tick_size": 10},
    "M0":  {"lot_value": 10, "tick_size": 1},
    "Y0":  {"lot_value": 10, "tick_size": 2},
    "C0":  {"lot_value": 10, "tick_size": 1},
    "SR0": {"lot_value": 10, "tick_size": 1},
    "CF0": {"lot_value": 5, "tick_size": 5},
}

# ---------- 数据设置 ----------
DATA_PARAMS = {
    "lookback_days": 500,       # 获取近 500 个交易日
    "cache_dir":     "data_cache",
    "cache_hours":   4,         # 缓存有效期（小时）
}

# ---------- 回测设置 ----------
BACKTEST_PARAMS = {
    "start_date": "2023-01-01",
    "end_date":   "2024-12-31",
}

# ---------- 本地模拟盘设置 ----------
PAPER_PARAMS = {
    "enabled": True,
    "start_date": "2024-01-01",
    "end_date": None,
    "storage_dir": "paper_trading",
    "max_daily_loss_pct": 0.03,
}

# ---------- 组合准入设置 ----------
# rolling_oos 只使用交易日之前的已平仓记录判断当日是否允许开仓，避免固定黑名单过拟合。
PORTFOLIO_FILTER_PARAMS = {
    "enabled": True,
    "mode": "rolling_oos",
    "lookback_days": 365,
    "min_trades": 2,
    "min_net_pnl": 0,
    "min_win_rate": 35,
    "min_profit_factor": 1.0,
    "blocked_symbols": [],
}
