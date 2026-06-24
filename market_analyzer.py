# ============================================================
#  市场状态分析模块
#  客观判断每个品种当前处于哪种状态，给出明确的操作建议
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from config import STATE_THRESHOLDS
from indicators import add_all_indicators, get_latest_row


# ─────────────────────────────────────────────
#  状态枚举
# ─────────────────────────────────────────────

class MarketState:
    OBSERVE      = "⚪ 观望"        # 无明确信号，等待
    LIGHT_LONG   = "🟡 轻仓做多"   # 短线多头信号，小仓
    LIGHT_SHORT  = "🟡 轻仓做空"   # 短线空头信号，小仓
    TREND_LONG   = "🟢 趋势做多"   # 强趋势上行，中等仓位
    TREND_SHORT  = "🔴 趋势做空"   # 强趋势下行，中等仓位


@dataclass
class AnalysisResult:
    symbol:    str
    state:     str
    score:     float
    direction: str          # "多" / "空" / "中性"
    reason:    str          # 文字描述判断依据
    stop_loss: float        # 参考止损价
    target:    float        # 参考止盈价
    entry:     float        # 参考入场价（当前收盘）
    atr:       float


# ─────────────────────────────────────────────
#  状态判断
# ─────────────────────────────────────────────

def _build_reason(row: pd.Series, state: str) -> str:
    """生成文字说明。"""
    parts = []

    # MA 排列
    ma5, ma20, ma60 = row.get("MA5"), row.get("MA20"), row.get("MA60")
    if pd.notna(ma5) and pd.notna(ma20):
        if ma5 > ma20:
            parts.append("MA5>MA20(多头排列)")
        else:
            parts.append("MA5<MA20(空头排列)")

    # RSI
    rsi = row.get("RSI")
    if pd.notna(rsi):
        if rsi > 70:
            parts.append(f"RSI={rsi:.0f}(超买)")
        elif rsi < 30:
            parts.append(f"RSI={rsi:.0f}(超卖)")
        else:
            parts.append(f"RSI={rsi:.0f}")

    # MACD
    dif, dea = row.get("DIF"), row.get("DEA")
    if pd.notna(dif) and pd.notna(dea):
        cross = "金叉" if dif > dea else "死叉"
        parts.append(f"MACD{cross}")

    # ADX
    adx = row.get("ADX")
    if pd.notna(adx):
        if adx > 40:
            parts.append(f"ADX={adx:.0f}(强趋势)")
        elif adx > 25:
            parts.append(f"ADX={adx:.0f}(有趋势)")
        else:
            parts.append(f"ADX={adx:.0f}(震荡)")

    return "；".join(parts)


def analyze_symbol(
    symbol: str,
    df: pd.DataFrame,
    score: float,
    direction: str,
) -> AnalysisResult:
    """
    根据综合评分 + 方向 + 指标细节，判断市场状态。

    Parameters
    ----------
    symbol    : 品种代码
    df        : 包含技术指标的 DataFrame
    score     : variety_selector 给出的综合分（0-100）
    direction : "多" 或 "空"
    """
    df_ind = add_all_indicators(df)
    row    = get_latest_row(df_ind)

    close = float(row["close"])
    atr   = float(row.get("ATR", close * 0.015))
    rsi   = float(row.get("RSI", 50))
    dif   = float(row.get("DIF", 0))
    dea   = float(row.get("DEA", 0))
    adx   = float(row.get("ADX", 0))
    ma5   = float(row.get("MA5", close))
    ma20  = float(row.get("MA20", close))

    # ── 布林带突破检测 ──
    bb_upper = float(row.get("BB_UPPER", close * 1.02))
    bb_lower = float(row.get("BB_LOWER", close * 0.98))
    bb_break_up   = close > bb_upper
    bb_break_down = close < bb_lower

    # ── 成交量放大检测 ──
    vol    = float(row.get("volume", 0))
    volma5 = float(row.get("VOL_MA5", vol))
    vol_surge = (volma5 > 0) and (vol > volma5 * 1.2)

    # ── 状态判断逻辑 ──
    if score >= STATE_THRESHOLDS["trend_long"] and direction == "多":
        # 强趋势做多：均线多头 + ADX 强 + MACD 金叉 + 量放大
        if adx > 25 and dif > dea and vol_surge:
            state = MarketState.TREND_LONG
        else:
            state = MarketState.LIGHT_LONG

    elif score >= STATE_THRESHOLDS["trend_short"] and direction == "空":
        if adx > 25 and dif < dea and vol_surge:
            state = MarketState.TREND_SHORT
        else:
            state = MarketState.LIGHT_SHORT

    elif score >= STATE_THRESHOLDS["light_trade"]:
        # 中等评分：根据方向选轻仓
        if direction == "多" and dif > dea:
            state = MarketState.LIGHT_LONG
        elif direction == "空" and dif < dea:
            state = MarketState.LIGHT_SHORT
        else:
            state = MarketState.OBSERVE

    else:
        # 低分 → 观望
        # 特例：RSI 超卖/超买时布林突破给短线机会
        if bb_break_up and rsi < 70 and direction == "多":
            state = MarketState.LIGHT_LONG
        elif bb_break_down and rsi > 30 and direction == "空":
            state = MarketState.LIGHT_SHORT
        else:
            state = MarketState.OBSERVE

    # ── 止损止盈 ──
    from config import RISK_PARAMS
    sl_mult = RISK_PARAMS["atr_stop_mult"]
    tp_mult = RISK_PARAMS["atr_target_mult"]

    if "多" in state:
        stop_loss = close - sl_mult * atr
        target    = close + tp_mult * atr
    elif "空" in state:
        stop_loss = close + sl_mult * atr
        target    = close - tp_mult * atr
    else:
        stop_loss = close - sl_mult * atr
        target    = close + tp_mult * atr

    reason = _build_reason(row, state)

    return AnalysisResult(
        symbol    = symbol,
        state     = state,
        score     = score,
        direction = direction,
        reason    = reason,
        stop_loss = round(stop_loss, 1),
        target    = round(target, 1),
        entry     = round(close, 1),
        atr       = round(atr, 1),
    )


def analyze_all(
    all_data: dict[str, pd.DataFrame],
    rank_df:  pd.DataFrame,
) -> list[AnalysisResult]:
    """
    分析所有品种状态，返回 AnalysisResult 列表（按评分降序）。
    """
    results = []
    score_map = dict(zip(rank_df["symbol"], rank_df["score"]))
    dir_map   = dict(zip(rank_df["symbol"], rank_df["direction"]))

    for symbol, df in all_data.items():
        if len(df) < 60:
            continue
        score = score_map.get(symbol, 0)
        direction = dir_map.get(symbol, "多")
        try:
            result = analyze_symbol(symbol, df, score, direction)
            results.append(result)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"分析 {symbol} 异常: {e}")

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def results_to_dataframe(results: list[AnalysisResult]) -> pd.DataFrame:
    """将 AnalysisResult 列表转为 DataFrame，便于展示。"""
    from config import ALL_SYMBOLS, SYMBOL_SECTOR
    rows = []
    for r in results:
        rows.append({
            "品种": f"{ALL_SYMBOLS.get(r.symbol, r.symbol)}({r.symbol})",
            "板块": SYMBOL_SECTOR.get(r.symbol, ""),
            "综合评分": r.score,
            "市场状态": r.state,
            "参考入场": r.entry,
            "参考止损": r.stop_loss,
            "参考止盈": r.target,
            "ATR":      r.atr,
            "判断依据": r.reason,
        })
    return pd.DataFrame(rows)
