# ============================================================
#  市场状态分析模块（v2）
#  整合：价格结构 + K线密度 + 量能 + 持仓量 → 四维综合判断
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from config import STATE_THRESHOLDS, RISK_PARAMS
from indicators import add_all_indicators, get_latest_row
from structure_analyzer import analyze_structure, StructureState
from volume_oi_analyzer import analyze_vol_oi, VolOIResult
from kline_density import analyze_density, DensityResult


# ─────────────────────────────────────────────
#  状态枚举
# ─────────────────────────────────────────────

class MarketState:
    OBSERVE      = "⚪ 观望"
    LIGHT_LONG   = "🟡 轻仓做多"
    LIGHT_SHORT  = "🟡 轻仓做空"
    TREND_LONG   = "🟢 趋势做多"
    TREND_SHORT  = "🔴 趋势做空"


# ─────────────────────────────────────────────
#  四维分析结果
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    symbol:    str
    name:      str

    # ── 核心状态 ──
    state:     str            # MarketState 中的一个
    score:     float          # 品种综合评分

    # ── 价格结构 ──
    structure: StructureState

    direction:   str          # "多" / "空"

    # ── 量价持仓 ──
    vol_oi: VolOIResult

    # ── K线密度 ──
    density: DensityResult

    # ── 交易参数 ──
    entry:     float
    stop_loss: float
    target:    float
    atr:       float

    # ── 综合描述（自然语言）──
    full_description: str


# ─────────────────────────────────────────────
#  自然语言行情描述生成
# ─────────────────────────────────────────────

def _build_full_description(
    name:      str,
    state:     str,
    structure: StructureState,
    row:       pd.Series,
    vol_oi:    VolOIResult,
    density:   "DensityResult",
    entry:     float,
    stop:      float,
    target:    float,
) -> str:
    """生成完整的中文行情描述，涵盖五个维度。"""
    lines = []

    # ① K线密度（最先看，决定是否参与）
    density_warn = ""
    if not density.tradeable:
        density_warn = " ⚠️ 当前密度过高，建议以下分析仅供参考，操作需谨慎"
    lines.append(f"【密度】{density.description}{density_warn}")

    # ② 价格结构
    lines.append(f"【结构】{structure.description}")
    lines.append(
        f"  支撑：{structure.support:.1f}  |  阻力：{structure.resistance:.1f}  "
        f"|  区间：{structure.recent_low:.1f}~{structure.recent_high:.1f}"
        f"（幅度 {structure.pivot_range_pct:.1f}%）"
    )

    # ③ 量价持仓
    lines.append(f"【量能】{vol_oi.vol_state.label}——{vol_oi.vol_state.description}")
    oi = vol_oi.oi_state
    if oi.code != "NO_DATA":
        lines.append(f"【持仓】{oi.label}——{oi.description}")
    else:
        lines.append("【持仓】暂无持仓量数据，建议参考交易所持仓排行")

    # ④ 操作建议
    action_map = {
        MarketState.OBSERVE:     "暂不操作，等待信号明确",
        MarketState.LIGHT_LONG:  f"轻仓做多，入场约 {entry:.1f}，止损 {stop:.1f}，目标 {target:.1f}",
        MarketState.LIGHT_SHORT: f"轻仓做空，入场约 {entry:.1f}，止损 {stop:.1f}，目标 {target:.1f}",
        MarketState.TREND_LONG:  f"趋势做多，入场约 {entry:.1f}，止损 {stop:.1f}（2xATR），目标 {target:.1f}（3xATR）",
        MarketState.TREND_SHORT: f"趋势做空，入场约 {entry:.1f}，止损 {stop:.1f}（2xATR），目标 {target:.1f}（3xATR）",
    }
    lines.append(f"【操作】{action_map.get(state, '观望')}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  状态判断逻辑
# ─────────────────────────────────────────────

def _determine_state(
    score:     float,
    direction: str,
    structure: StructureState,
    row:       pd.Series,
    vol_oi:    VolOIResult,
    density:   "DensityResult",
) -> str:
    """
    五维联合判断市场状态。
    优先级：K线密度 > 价格结构 > 量价持仓 > 技术指标 > 综合评分

    密度规则（最高优先级）：
      密度 >= 75 → 强制观望（拥挤行情不操作）
      密度 65-75 → 最高只允许轻仓（不允许趋势仓位）
    """
    # ── 密度过滤（最优先）──
    if density.score >= 75:
        return MarketState.OBSERVE
    downgrade = density.score >= 65   # 高密度时不允许趋势级别仓位

    sub   = structure.sub_state
    trend = structure.trend
    vs    = vol_oi.combined_signal
    vc    = vol_oi.combined_score

    # ── 突破状态优先 ──
    if sub == "BREAKOUT_UP" and (vs in ("strong_bull", "bull")):
        raw = MarketState.TREND_LONG if score >= 55 else MarketState.LIGHT_LONG
        return MarketState.LIGHT_LONG if (downgrade and raw == MarketState.TREND_LONG) else raw
    if sub == "BREAKOUT_DN" and (vs in ("strong_bear", "bear")):
        raw = MarketState.TREND_SHORT if score >= 55 else MarketState.LIGHT_SHORT
        return MarketState.LIGHT_SHORT if (downgrade and raw == MarketState.TREND_SHORT) else raw

    # ── 趋势+回踩=做多机会 ──
    if sub == "PULLBACK_UP" and trend == "UPTREND":
        if vs in ("strong_bull", "bull", "neutral"):
            return MarketState.LIGHT_LONG
        if vs == "strong_bull" and not downgrade:
            return MarketState.TREND_LONG

    # ── 趋势+反弹=做空机会 ──
    if sub == "PULLBACK_DN" and trend == "DOWNTREND":
        if vs in ("strong_bear", "bear", "neutral"):
            return MarketState.LIGHT_SHORT
        if vs == "strong_bear" and not downgrade:
            return MarketState.TREND_SHORT

    # ── 强结构 + 量价持仓同向 ──
    if score >= STATE_THRESHOLDS["trend_long"] and not downgrade:
        if direction == "多" and trend == "UPTREND" and vs in ("strong_bull", "bull"):
            return MarketState.TREND_LONG
        if direction == "空" and trend == "DOWNTREND" and vs in ("strong_bear", "bear"):
            return MarketState.TREND_SHORT

    # ── 中等信号 → 轻仓 ──
    if score >= STATE_THRESHOLDS["light_trade"]:
        if direction == "多" and trend != "DOWNTREND":
            return MarketState.LIGHT_LONG
        if direction == "空" and trend != "UPTREND":
            return MarketState.LIGHT_SHORT

    return MarketState.OBSERVE


# ─────────────────────────────────────────────
#  主分析接口
# ─────────────────────────────────────────────

def analyze_symbol(
    symbol:    str,
    df:        pd.DataFrame,
    score:     float,
    direction: str,
) -> AnalysisResult:
    """四维分析单个品种，返回完整 AnalysisResult。"""
    from config import ALL_SYMBOLS
    name = ALL_SYMBOLS.get(symbol, symbol)

    df_ind = add_all_indicators(df)
    row    = get_latest_row(df_ind)

    close = float(row["close"])
    atr   = float(row.get("ATR", close * 0.015))

    structure = analyze_structure(df_ind)
    vol_oi    = analyze_vol_oi(df_ind)
    density   = analyze_density(df_ind)
    state     = _determine_state(score, direction, structure, row, vol_oi, density)

    sl = RISK_PARAMS["atr_stop_mult"]
    tp = RISK_PARAMS["atr_target_mult"]
    if "多" in state:
        stop   = min(close - sl * atr, structure.support - 0.5 * atr)
        target = close + tp * atr
    elif "空" in state:
        stop   = max(close + sl * atr, structure.resistance + 0.5 * atr)
        target = close - tp * atr
    else:
        stop   = close - sl * atr
        target = close + tp * atr

    desc = _build_full_description(
        name, state, structure, row, vol_oi, density,
        entry=close, stop=round(stop, 1), target=round(target, 1),
    )

    return AnalysisResult(
        symbol=symbol, name=name, state=state, score=score,
        structure=structure, direction=direction,
        vol_oi=vol_oi, density=density,
        entry=round(close, 1), stop_loss=round(stop, 1),
        target=round(target, 1), atr=round(atr, 1),
        full_description=desc,
    )


def analyze_all(
    all_data: "dict[str, pd.DataFrame]",
    rank_df:  pd.DataFrame,
) -> "list[AnalysisResult]":
    """分析所有品种，返回按评分降序的列表。"""
    import logging
    logger = logging.getLogger(__name__)

    score_map = dict(zip(rank_df["symbol"], rank_df["score"]))
    dir_map   = dict(zip(rank_df["symbol"], rank_df["direction"]))
    results   = []

    for sym, df in all_data.items():
        if len(df) < 60:
            continue
        try:
            r = analyze_symbol(sym, df, score_map.get(sym, 0), dir_map.get(sym, "多"))
            results.append(r)
        except Exception as e:
            logger.warning(f"分析 {sym} 异常: {e}")

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def results_to_dataframe(results: "list[AnalysisResult]") -> pd.DataFrame:
    """转为简洁 DataFrame（用于仪表板表格）。"""
    rows = []
    for r in results:
        rows.append({
            "品种":     f"{r.name}({r.symbol})",
            "板块":     __import__("config").SYMBOL_SECTOR.get(r.symbol, ""),
            "综合评分": r.score,
            "市场状态": r.state,
            "K线密度":  f"{r.density.score:.0f} {r.density.label}",
            "价格结构": r.structure.trend + "/" + r.structure.sub_state,
            "量能":     r.vol_oi.vol_state.label,
            "持仓信号": r.vol_oi.oi_state.label,
            "参考入场": r.entry,
            "止损":     r.stop_loss,
            "止盈":     r.target,
            "ATR":      r.atr,
        })
    return pd.DataFrame(rows)


def get_detail_text(result: AnalysisResult) -> str:
    """返回单品种完整分析文本（用于仪表板详情面板）。"""
    return (
        f"━━ {result.name}({result.symbol}) 综合分析 ━━\n"
        f"综合评分：{result.score:.1f} / 100\n"
        f"当前状态：{result.state}\n\n"
        + result.full_description
    )
