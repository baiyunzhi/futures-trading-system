# ============================================================
#  价格结构分析模块
#  核心：识别图形高低点 → 判断价格所处结构状态
# ============================================================
"""
高低点认定标准
──────────────
摆动高点（Swing High）：某根 K 线的最高价，高于其左侧 N 根和右侧 N 根 K 线的最高价。
摆动低点（Swing Low）：某根 K 线的最低价，低于其左侧 N 根和右侧 N 根 K 线的最低价。

N 默认 = 5（短周期，适合日线短线交易）。
N = 10 识别中周期关键位（用于判断主要支撑/阻力）。

有效性附加条件（避免密集区假高低点）：
  - 相邻两个同向极值点之间的价格振幅 >= 1×ATR(14)
  - 相邻两个同向极值点之间至少间隔 3 根 K 线

价格结构状态（6种）：
  UPTREND      上升趋势（近期 HH+HL，高点和低点均抬升）
  DOWNTREND    下降趋势（近期 LH+LL，高点和低点均下移）
  RANGE        震荡区间（高点无明显抬升，低点无明显下移）
  BREAKOUT_UP  向上突破（价格突破近期区间上沿）
  BREAKOUT_DN  向下突破（价格跌破近期区间下沿）
  PULLBACK_UP  上升趋势回踩（趋势向上但价格回落测试支撑）
  PULLBACK_DN  下降趋势反弹（趋势向下但价格反弹测试阻力）
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────

@dataclass
class PivotPoint:
    idx:       int             # 数组下标
    date:      pd.Timestamp
    price:     float
    kind:      str             # "HIGH" / "LOW"
    strength:  str             # "MAJOR"（中周期）/ "MINOR"（短周期）


@dataclass
class StructureState:
    trend:      str            # UPTREND / DOWNTREND / RANGE
    sub_state:  str            # BREAKOUT_UP / BREAKOUT_DN / PULLBACK_UP / PULLBACK_DN / NORMAL
    description: str           # 中文描述

    # 关键价位
    recent_high: float         # 近期有效高点
    recent_low:  float         # 近期有效低点
    support:     float         # 当前最近支撑
    resistance:  float         # 当前最近阻力
    pivot_range_pct: float     # (高-低)/低 * 100，区间宽度

    # 高低点序列（最近 3 个）
    highs: list[PivotPoint] = field(default_factory=list)
    lows:  list[PivotPoint] = field(default_factory=list)


# ─────────────────────────────────────────────
#  高低点识别
# ─────────────────────────────────────────────

def find_pivots(
    df:       pd.DataFrame,
    n:        int   = 5,       # 左右各 N 根
    min_atr_dist: float = 1.0, # 相邻同向极值最小间距（倍ATR）
    strength: str   = "MINOR",
) -> tuple[list[PivotPoint], list[PivotPoint]]:
    """
    识别摆动高点和低点。

    Parameters
    ----------
    df           : 含 high / low / close / ATR 列的 DataFrame
    n            : 左右各比较 n 根 K 线
    min_atr_dist : 相邻同向极值的最小价格距离（以 ATR 的倍数）
    strength     : "MINOR" 或 "MAJOR"

    Returns
    -------
    (highs, lows): 两个 PivotPoint 列表，按时间升序
    """
    highs_raw: list[PivotPoint] = []
    lows_raw:  list[PivotPoint] = []

    high_arr  = df["high"].values
    low_arr   = df["low"].values
    atr_arr   = df["ATR"].fillna(df["close"] * 0.015).values
    dates     = df["date"].values

    for i in range(n, len(df) - n):
        atr_i = atr_arr[i]

        # ── 摆动高点 ──
        window_highs = high_arr[i - n: i + n + 1]
        if high_arr[i] == window_highs.max():
            highs_raw.append(PivotPoint(i, pd.Timestamp(dates[i]), float(high_arr[i]), "HIGH", strength))

        # ── 摆动低点 ──
        window_lows = low_arr[i - n: i + n + 1]
        if low_arr[i] == window_lows.min():
            lows_raw.append(PivotPoint(i, pd.Timestamp(dates[i]), float(low_arr[i]), "LOW", strength))

    # ── 去重：相邻同向极值需满足最小间距 ──
    def filter_pivots(raw: list[PivotPoint], kind: str) -> list[PivotPoint]:
        filtered = []
        for p in raw:
            if not filtered:
                filtered.append(p)
                continue
            prev = filtered[-1]
            atr_ref = atr_arr[p.idx]
            price_dist = abs(p.price - prev.price)
            bar_dist   = p.idx - prev.idx

            if bar_dist < 3:
                # 太近：只保留更极端的那个
                if kind == "HIGH" and p.price > prev.price:
                    filtered[-1] = p
                elif kind == "LOW" and p.price < prev.price:
                    filtered[-1] = p
            elif price_dist < min_atr_dist * atr_ref:
                # 价格差异太小：也只保留更极端的
                if kind == "HIGH" and p.price > prev.price:
                    filtered[-1] = p
                elif kind == "LOW" and p.price < prev.price:
                    filtered[-1] = p
            else:
                filtered.append(p)
        return filtered

    highs = filter_pivots(highs_raw, "HIGH")
    lows  = filter_pivots(lows_raw, "LOW")
    return highs, lows


# ─────────────────────────────────────────────
#  价格结构判断
# ─────────────────────────────────────────────

def _classify_trend(highs: list[PivotPoint], lows: list[PivotPoint]) -> str:
    """
    根据近3个高点和低点序列判断趋势结构。

    上升趋势：高点抬升 且 低点抬升（HH + HL）
    下降趋势：高点下移 且 低点下移（LH + LL）
    震荡：其他情况
    """
    if len(highs) >= 2 and len(lows) >= 2:
        h_up = highs[-1].price > highs[-2].price   # 高点抬升
        l_up = lows[-1].price  > lows[-2].price    # 低点抬升
        h_dn = highs[-1].price < highs[-2].price   # 高点下移
        l_dn = lows[-1].price  < lows[-2].price    # 低点下移

        if h_up and l_up:
            return "UPTREND"
        if h_dn and l_dn:
            return "DOWNTREND"

    return "RANGE"


def _classify_sub_state(
    df:        pd.DataFrame,
    trend:     str,
    highs:     list[PivotPoint],
    lows:      list[PivotPoint],
) -> str:
    """
    在已知趋势的前提下，判断当前价格处于哪个子状态。
    """
    if not highs or not lows:
        return "NORMAL"

    close      = df["close"].iloc[-1]
    atr        = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else close * 0.015
    recent_h   = highs[-1].price
    recent_l   = lows[-1].price
    range_size = recent_h - recent_l

    # 突破判断（价格超出近期高低点区间）
    if close > recent_h + 0.1 * atr:
        return "BREAKOUT_UP"
    if close < recent_l - 0.1 * atr:
        return "BREAKOUT_DN"

    # 回踩/反弹判断
    if trend == "UPTREND":
        # 上升趋势中，价格回落到近期低点附近（±1.5 ATR）→ 回踩支撑
        near_support = abs(close - recent_l) < 1.5 * atr
        if near_support:
            return "PULLBACK_UP"

    if trend == "DOWNTREND":
        # 下降趋势中，价格反弹到近期高点附近 → 反弹测阻
        near_resistance = abs(close - recent_h) < 1.5 * atr
        if near_resistance:
            return "PULLBACK_DN"

    return "NORMAL"


def _build_description(
    trend:    str,
    sub:      str,
    close:    float,
    highs:    list[PivotPoint],
    lows:     list[PivotPoint],
    support:  float,
    resist:   float,
) -> str:
    """生成人类可读的中文价格结构描述。"""

    trend_zh = {
        "UPTREND":   "上升趋势",
        "DOWNTREND": "下降趋势",
        "RANGE":     "震荡区间",
    }.get(trend, "未明")

    # 描述高低点序列
    hh_hl = hl_info = ""
    if len(highs) >= 2:
        h_dir = "逐步抬升（HH）" if highs[-1].price > highs[-2].price else "逐步下移（LH）"
        hh_hl = f"近期高点{h_dir}"
    if len(lows) >= 2:
        l_dir = "逐步抬升（HL）" if lows[-1].price > lows[-2].price else "逐步下移（LL）"
        hl_info = f"近期低点{l_dir}"

    base = f"当前处于{trend_zh}"
    if hh_hl and hl_info:
        base += f"，{hh_hl}，{hl_info}"

    sub_desc = {
        "BREAKOUT_UP":  f"，价格向上突破近期高点{highs[-1].price:.1f}，关注回踩确认",
        "BREAKOUT_DN":  f"，价格向下跌破近期低点{lows[-1].price:.1f}，空头信号强",
        "PULLBACK_UP":  f"，价格回踩支撑区{support:.1f}附近，若止跌企稳可尝试做多",
        "PULLBACK_DN":  f"，价格反弹至阻力区{resist:.1f}附近，若受阻回落可尝试做空",
        "NORMAL":       f"，当前价{close:.1f}处于区间中段",
    }.get(sub, "")

    return base + sub_desc


# ─────────────────────────────────────────────
#  主接口
# ─────────────────────────────────────────────

def analyze_structure(
    df:      pd.DataFrame,
    n_minor: int = 5,    # 短周期摆动（日线）
    n_major: int = 10,   # 中周期摆动
) -> StructureState:
    """
    分析 df 的价格结构，返回 StructureState。
    df 需含 high / low / close / ATR / date 列。
    """
    if len(df) < (n_major * 2 + 5):
        return StructureState(
            trend="RANGE", sub_state="NORMAL",
            description="数据不足，无法判断结构",
            recent_high=float(df["high"].max()),
            recent_low=float(df["low"].min()),
            support=float(df["low"].min()),
            resistance=float(df["high"].max()),
            pivot_range_pct=0,
        )

    # 识别中周期高低点（用于趋势判断）
    highs_maj, lows_maj = find_pivots(df, n=n_major, min_atr_dist=1.5, strength="MAJOR")

    # 识别短周期高低点（用于细节分析）
    highs_min, lows_min = find_pivots(df, n=n_minor, min_atr_dist=0.8, strength="MINOR")

    # 用中周期极值判断大趋势
    trend = _classify_trend(highs_maj, lows_maj)

    # 取最近几个小周期高低点做子状态判断
    close = float(df["close"].iloc[-1])
    sub   = _classify_sub_state(df, trend, highs_min, lows_min)

    # 支撑阻力：最近低点/高点
    recent_low  = lows_min[-1].price  if lows_min  else float(df["low"].min())
    recent_high = highs_min[-1].price if highs_min else float(df["high"].max())

    # 计算离当前价最近的支撑（低于收盘）和阻力（高于收盘）
    all_lows  = sorted([p.price for p in lows_min  if p.price < close], reverse=True)
    all_highs = sorted([p.price for p in highs_min if p.price > close])
    support    = all_lows[0]  if all_lows  else recent_low
    resistance = all_highs[0] if all_highs else recent_high

    range_pct = (recent_high - recent_low) / recent_low * 100 if recent_low > 0 else 0

    desc = _build_description(trend, sub, close, highs_min[-3:], lows_min[-3:], support, resistance)

    return StructureState(
        trend       = trend,
        sub_state   = sub,
        description = desc,
        recent_high = round(recent_high, 1),
        recent_low  = round(recent_low, 1),
        support     = round(support, 1),
        resistance  = round(resistance, 1),
        pivot_range_pct = round(range_pct, 2),
        highs = highs_min[-5:],   # 最近 5 个短周期高点
        lows  = lows_min[-5:],    # 最近 5 个短周期低点
    )


def get_key_levels(state: StructureState, n: int = 3) -> list[dict]:
    """
    提取关键价位列表（用于图表标注和风控参考）。
    返回 [{"price": ..., "label": ..., "kind": "support"/"resistance"}, ...]
    """
    levels = []
    for p in state.highs[-n:]:
        levels.append({"price": p.price, "label": f"阻力 {p.price:.0f}",
                       "kind": "resistance", "date": p.date})
    for p in state.lows[-n:]:
        levels.append({"price": p.price, "label": f"支撑 {p.price:.0f}",
                       "kind": "support", "date": p.date})
    return sorted(levels, key=lambda x: x["price"], reverse=True)
