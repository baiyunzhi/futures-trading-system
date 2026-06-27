from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from config import ALL_SYMBOLS, CONTRACT_SPECS, RISK_PARAMS, SYMBOL_SECTOR
from indicators import add_all_indicators
from kline_density import analyze_density
from signal_generator import Signal
from structure_analyzer import StructureState, analyze_structure
from portfolio_filter import SymbolEligibility


@dataclass(frozen=True)
class TradeDecision:
    symbol: str
    name: str
    sector: str
    tradable: bool
    regime: str
    direction: str
    strategy: str
    entry: float
    stop_loss: float
    target: float
    risk_amount: float
    lots: int
    score: float
    invalidation: str
    execution_note: str
    structure: StructureState


def classify_regime(df: pd.DataFrame, structure: StructureState) -> str:
    if df.empty:
        return "NO_DATA"
    row = df.iloc[-1]
    close = float(row["close"])
    atr = float(row.get("ATR", close * 0.015))
    density = analyze_density(df)
    range_pct = float(structure.pivot_range_pct)

    if density.score >= 75 or range_pct < 1.2:
        return "CHOP"
    if structure.sub_state in ("BREAKOUT_UP", "BREAKOUT_DN"):
        return "BREAKOUT"
    if structure.trend in ("UPTREND", "DOWNTREND"):
        return "TREND"
    if range_pct >= max(2.0, atr / close * 100 * 2.5) and density.score < 70:
        return "RANGE"
    return "CHOP"


def select_strategy(regime: str, structure: StructureState) -> tuple[str, str]:
    if regime == "TREND":
        direction = "多" if structure.trend == "UPTREND" else "空"
        return "趋势跟随", direction
    if regime == "BREAKOUT":
        direction = "多" if structure.sub_state == "BREAKOUT_UP" else "空"
        return "突破回踩", direction
    if regime == "RANGE":
        return "区间反转", "双向等待"
    return "禁止交易", "观望"


def strategy_for_regime(regime: str) -> Callable[[str, pd.DataFrame], list[Signal]] | None:
    if regime == "TREND":
        import strategy_structure as strategy
        return strategy.generate_signals
    return None


def calc_risk_first_lots(
    capital: float,
    entry: float,
    stop_loss: float,
    lot_value: float,
    risk_pct: float | None = None,
) -> tuple[float, int]:
    risk_pct = RISK_PARAMS["max_risk_per_trade"] if risk_pct is None else risk_pct
    risk_amount = capital * risk_pct
    risk_per_lot = abs(entry - stop_loss) * lot_value
    if entry <= 0 or stop_loss <= 0 or risk_per_lot <= 0:
        return risk_amount, 0
    return risk_amount, max(0, int(risk_amount / risk_per_lot))


def _brackets(close: float, atr: float, structure: StructureState, direction: str, regime: str) -> tuple[float, float]:
    if direction == "多":
        if regime == "TREND":
            stop = min(structure.support, structure.recent_low) - 0.5 * atr
            target = close + max((close - stop) * 2.0, 3.0 * atr)
        elif regime == "BREAKOUT":
            stop = structure.recent_high - 0.5 * atr
            target = close + max((close - stop) * 2.0, structure.pivot_range_pct / 100 * close)
        else:
            stop = structure.recent_low - atr
            target = (structure.recent_high + structure.recent_low) / 2
        if stop >= close:
            stop = close - RISK_PARAMS["atr_stop_mult"] * atr
            target = close + RISK_PARAMS["atr_target_mult"] * atr
    elif direction == "空":
        if regime == "TREND":
            stop = max(structure.resistance, structure.recent_high) + 0.5 * atr
            target = close - max((stop - close) * 2.0, 3.0 * atr)
        elif regime == "BREAKOUT":
            stop = structure.recent_low + 0.5 * atr
            target = close - max((stop - close) * 2.0, structure.pivot_range_pct / 100 * close)
        else:
            stop = structure.recent_high + atr
            target = (structure.recent_high + structure.recent_low) / 2
        if stop <= close:
            stop = close + RISK_PARAMS["atr_stop_mult"] * atr
            target = close - RISK_PARAMS["atr_target_mult"] * atr
    else:
        stop = close - 2 * atr
        target = close + 3 * atr
    return round(float(stop), 2), round(float(target), 2)


def build_trade_decision(
    symbol: str,
    df: pd.DataFrame,
    capital: float = RISK_PARAMS["capital"],
    eligibility: SymbolEligibility | None = None,
) -> TradeDecision:
    df_ind = add_all_indicators(df) if "ATR" not in df.columns else df.copy()
    if len(df_ind) < 80:
        structure = analyze_structure(df_ind)
        return TradeDecision(
            symbol, ALL_SYMBOLS.get(symbol, symbol), SYMBOL_SECTOR.get(symbol, ""),
            False, "NO_DATA", "观望", "禁止交易", 0, 0, 0, 0, 0, 0,
            "数据不足", "等待至少 80 根有效K线", structure,
        )

    structure = analyze_structure(df_ind)
    regime = classify_regime(df_ind, structure)
    strategy, direction = select_strategy(regime, structure)
    allowed_by_filter = True if eligibility is None else eligibility.allowed
    row = df_ind.iloc[-1]
    close = float(row["close"])
    atr = float(row.get("ATR", close * 0.015))
    stop, target = _brackets(close, atr, structure, direction, regime)
    lot_value = float(CONTRACT_SPECS.get(symbol, {}).get("lot_value", RISK_PARAMS["default_lot_value"]))
    risk_amount, lots = calc_risk_first_lots(capital, close, stop, lot_value)

    reward = abs(target - close)
    risk = abs(close - stop)
    rr = reward / risk if risk > 0 else 0
    density = analyze_density(df_ind)
    tradable = allowed_by_filter and regime not in ("CHOP", "NO_DATA") and lots > 0 and rr >= 1.8 and density.score < 75
    score = 0.0
    if tradable:
        score = min(100.0, 45 + rr * 12 + max(0, 75 - density.score) * 0.4 + min(15, structure.pivot_range_pct))

    invalidation = eligibility.reason if eligibility is not None and not eligibility.allowed else {
        "TREND": "多头跌破最近结构低点 / 空头突破最近结构高点",
        "BREAKOUT": "价格重新回到突破区间内部",
        "RANGE": "有效突破区间边界，区间策略失效",
    }.get(regime, "结构混乱或密度过高，禁止开仓")
    execution_note = "滚动样本外准入未通过：暂停交易，只观察不下单" if eligibility is not None and not eligibility.allowed else (
        "等待策略触发信号；先按止损距离计算手数，再允许开仓" if tradable else "不满足可交易条件，保持观望"
    )
    if eligibility is not None and not eligibility.allowed:
        strategy = f"{strategy}/滚动禁入"

    return TradeDecision(
        symbol=symbol,
        name=ALL_SYMBOLS.get(symbol, symbol),
        sector=SYMBOL_SECTOR.get(symbol, ""),
        tradable=tradable,
        regime=regime,
        direction=direction,
        strategy=strategy,
        entry=round(close, 2),
        stop_loss=stop,
        target=target,
        risk_amount=round(risk_amount, 2),
        lots=lots,
        score=round(score, 1),
        invalidation=invalidation,
        execution_note=execution_note,
        structure=structure,
    )


def build_all_trade_decisions(
    all_data: dict[str, pd.DataFrame],
    capital: float = RISK_PARAMS["capital"],
    eligibility_snapshot: dict[str, SymbolEligibility] | None = None,
) -> list[TradeDecision]:
    decisions = []
    for symbol, df in all_data.items():
        if df is None or df.empty:
            continue
        try:
            decisions.append(build_trade_decision(symbol, df, capital, eligibility=(eligibility_snapshot or {}).get(symbol)))
        except Exception:
            continue
    return sorted(decisions, key=lambda d: (d.tradable, d.score), reverse=True)


def decisions_to_dataframe(decisions: list[TradeDecision]) -> pd.DataFrame:
    return pd.DataFrame([{
        "品种": f"{d.name}({d.symbol})",
        "板块": d.sector,
        "可交易": "是" if d.tradable else "否",
        "结构": d.regime,
        "方向": d.direction,
        "策略": d.strategy,
        "入场": d.entry,
        "止损": d.stop_loss,
        "目标": d.target,
        "单笔风险": d.risk_amount,
        "建议手数": d.lots,
        "评分": d.score,
        "失效条件": d.invalidation,
        "执行建议": d.execution_note,
    } for d in decisions])
