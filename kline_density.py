# ============================================================
#  K 线密度分析模块
#  量化"行情清晰度"：密度低=简洁趋势=容易操作；密度高=震荡拉锯=建议观望
# ============================================================
"""
三个核心指标：

1. 相邻K线重合度（Overlap Ratio）
   ─────────────────────────────────
   相邻两根K线高低点区间的重叠量 / 两根K线中较小那根的区间。
   重合度 = 0  → 完全不重叠（单边行情，最清晰）
   重合度 = 1  → 完全重叠（当前K线在前一根区间内，最拥挤）
   取近N根的均值作为区间评分。

2. 震荡指数（Choppiness Index, CI）
   ────────────────────────────────
   CI(n) = 100 × log10(ΣTR(1…n) / (HH(n) - LL(n))) / log10(n)

   原理：如果N根K线各自的波动都被用来推进价格方向，
         则CI趋近下限（趋势市）；
         如果N根K线反复震荡，总位移远小于总波动之和，
         则CI趋近上限（震荡市）。

   参考阈值（Fibonacci分割）：
     CI < 38.2  → 强趋势，行情极清晰
     38.2–61.8 → 中性
     CI > 61.8  → 震荡拉锯，不宜交易

3. K线实体率（Body Ratio）
   ────────────────────────
   实体 = |close - open|
   全幅 = high - low
   实体率 = 实体 / 全幅（排除十字星等无方向K线）
   高实体率 → 方向明确；低实体率 → 多空分歧

综合密度评分（0-100）：
  0–30   "简洁行情" 绿  → 趋势明确，适合交易
  30–55  "中性行情" 黄  → 可选择性参与
  55–75  "偏密集"  橙  → 谨慎，降低仓位
  75–100 "拥挤行情" 红  → 建议观望，等待密度下降
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


# ─────────────────────────────────────────────
#  参数
# ─────────────────────────────────────────────

CI_PERIOD       = 14     # 震荡指数计算周期
OVERLAP_LOOKBACK = 10    # 重合度均值窗口
BODY_LOOKBACK   = 10     # 实体率均值窗口
DIR_LOOKBACK    = 10     # 方向一致性窗口

CI_TREND_THRESH  = 38.2  # CI低于此值 → 强趋势
CI_CHOP_THRESH   = 61.8  # CI高于此值 → 震荡

# 综合密度评分分级
DENSITY_LEVELS = [
    (0,  30,  "简洁行情", "#26a69a", "行情清晰，适合操作"),
    (30, 55,  "中性行情", "#FFD700", "可选择性参与，注意节奏"),
    (55, 75,  "偏密集",  "#FFA500", "K线重叠较多，谨慎降仓"),
    (75, 101, "拥挤行情", "#ef5350", "高度震荡拉锯，建议观望"),
]


# ─────────────────────────────────────────────
#  指标计算
# ─────────────────────────────────────────────

def choppiness_index(df: pd.DataFrame, n: int = CI_PERIOD) -> pd.Series:
    """
    震荡指数 CI，范围约 0–100。
    值越高=越震荡；值越低=趋势越强。
    """
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    sum_tr = tr.rolling(n).sum()
    hh     = df["high"].rolling(n).max()
    ll     = df["low"].rolling(n).min()
    denom  = (hh - ll).replace(0, np.nan)

    ci = 100 * np.log10(sum_tr / denom) / np.log10(n)
    return ci.clip(0, 100)


def overlap_ratio_series(df: pd.DataFrame, lookback: int = OVERLAP_LOOKBACK) -> pd.Series:
    """
    计算每日相邻K线重合度，取滚动均值。
    重合度 0=完全不重叠（单边），1=完全包含（内包K线）。
    """
    overlap_vals = []
    high = df["high"].values
    low  = df["low"].values

    for i in range(len(df)):
        if i == 0:
            overlap_vals.append(np.nan)
            continue
        overlap = max(0.0, min(high[i], high[i-1]) - max(low[i], low[i-1]))
        bar_range = min(high[i] - low[i], high[i-1] - low[i-1])
        if bar_range <= 0:
            overlap_vals.append(1.0)
        else:
            overlap_vals.append(min(1.0, overlap / bar_range))

    raw = pd.Series(overlap_vals, index=df.index)
    return raw.rolling(lookback).mean()


def body_ratio_series(df: pd.DataFrame, lookback: int = BODY_LOOKBACK) -> pd.Series:
    """
    K线实体率（body / total range）的滚动均值。
    高实体率 → 方向明确；低实体率 → 多空拉锯，蜡烛影线长。
    返回值 0–1，越高越清晰。
    """
    body  = (df["close"] - df["open"]).abs()
    total = (df["high"] - df["low"]).replace(0, np.nan)
    ratio = (body / total).clip(0, 1)
    return ratio.rolling(lookback).mean()


def direction_consistency(df: pd.DataFrame, lookback: int = DIR_LOOKBACK) -> pd.Series:
    """
    近 N 根K线方向一致性：同向K线占比（0–1）。
    趋势市中该值接近1（比如连续8根阳线）；
    震荡市中该值接近0.5（多空交替）。
    """
    direction = np.sign(df["close"] - df["open"])   # 1=阳 -1=阴 0=十字
    # 主方向：该窗口内占多数的方向
    def _consistency(w):
        if len(w) == 0:
            return np.nan
        pos = (w == 1).sum()
        neg = (w == -1).sum()
        total = len(w)
        return max(pos, neg) / total if total > 0 else 0.5

    return direction.rolling(lookback).apply(_consistency, raw=True)


# ─────────────────────────────────────────────
#  综合密度评分
# ─────────────────────────────────────────────

def density_score_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算历史每日的密度评分，返回含各分项的 DataFrame。
    用于可视化历史密度趋势。
    """
    ci        = choppiness_index(df)
    overlap   = overlap_ratio_series(df)
    body      = body_ratio_series(df)
    direction = direction_consistency(df)

    # 各分项标准化为 0-100（越高=越密集）
    ci_score      = ci.clip(0, 100)                      # CI 本身即 0-100，高=密集
    overlap_score = (overlap.fillna(0.5) * 100).clip(0, 100)  # 重合度 0-1 → 0-100
    body_score    = ((1 - body.fillna(0.5)) * 100).clip(0, 100)  # 实体率低=拉锯=密集
    dir_score     = ((1 - direction.fillna(0.5)) * 100).clip(0, 100)  # 一致性低=混乱=密集

    # 加权合并
    composite = (
        ci_score      * 0.40 +
        overlap_score * 0.30 +
        body_score    * 0.15 +
        dir_score     * 0.15
    )

    return pd.DataFrame({
        "date":           df["date"],
        "density_score":  composite.round(1),
        "ci":             ci_score.round(1),
        "overlap":        overlap_score.round(1),
        "body_score":     body_score.round(1),
        "dir_score":      dir_score.round(1),
    })


# ─────────────────────────────────────────────
#  当前密度快照（用于实盘判断）
# ─────────────────────────────────────────────

@dataclass
class DensityResult:
    score:        float     # 0-100，越高越密集
    label:        str       # 简洁行情 / 中性行情 / 偏密集 / 拥挤行情
    color:        str       # 颜色代码
    description:  str       # 文字描述

    ci:           float     # 震荡指数
    ci_state:     str       # "趋势" / "中性" / "震荡"
    overlap:      float     # 近期K线平均重合度（0-1）
    body_ratio:   float     # 平均实体率（0-1）
    dir_consist:  float     # 方向一致性（0-1）

    tradeable:    bool      # 是否建议参与交易
    penalty:      float     # 对品种评分的惩罚分（0-35）


def analyze_density(df: pd.DataFrame) -> DensityResult:
    """
    计算当前时刻的密度快照，供 market_analyzer 调用。
    """
    if len(df) < CI_PERIOD + 5:
        return DensityResult(
            score=50, label="数据不足", color="#888", description="数据不足",
            ci=50, ci_state="中性", overlap=0.5, body_ratio=0.5,
            dir_consist=0.5, tradeable=True, penalty=0,
        )

    density_df = density_score_series(df)
    latest = density_df.dropna(subset=["density_score"])
    if latest.empty:
        return DensityResult(
            score=50, label="计算异常", color="#888", description="计算异常",
            ci=50, ci_state="中性", overlap=0.5, body_ratio=0.5,
            dir_consist=0.5, tradeable=True, penalty=0,
        )

    row = latest.iloc[-1]
    score     = float(row["density_score"])
    ci_val    = float(row["ci"])
    overlap   = float(row["overlap"]) / 100
    body      = 1 - float(row["body_score"]) / 100
    direction = 1 - float(row["dir_score"]) / 100

    # CI 文字状态
    if ci_val < CI_TREND_THRESH:
        ci_state = f"强趋势（CI={ci_val:.0f}<38.2）"
    elif ci_val > CI_CHOP_THRESH:
        ci_state = f"震荡拉锯（CI={ci_val:.0f}>61.8）"
    else:
        ci_state = f"中性（CI={ci_val:.0f}）"

    # 分级
    label, color, desc_base = "中性行情", "#FFD700", ""
    for lo, hi, lbl, clr, desc in DENSITY_LEVELS:
        if lo <= score < hi:
            label, color, desc_base = lbl, clr, desc
            break

    # 详细描述
    description = (
        f"K线密度评分 {score:.0f}/100 → {label}。{desc_base}。"
        f"震荡指数：{ci_state}；"
        f"相邻K线重合度：{overlap:.0%}；"
        f"平均实体率：{body:.0%}（实体比例{'高，方向明确' if body > 0.6 else '低，多空分歧'}）；"
        f"方向一致性：{direction:.0%}（近{DIR_LOOKBACK}根{'同向居多，趋势明显' if direction > 0.7 else '多空交替，震荡特征'}）"
    )

    tradeable = score < 65
    # 惩罚分：分数越高惩罚越重（线性映射 55-100 → 0-35）
    penalty = max(0.0, (score - 55) / 45 * 35) if score > 55 else 0.0

    return DensityResult(
        score       = round(score, 1),
        label       = label,
        color       = color,
        description = description,
        ci          = round(ci_val, 1),
        ci_state    = ci_state,
        overlap     = round(overlap, 3),
        body_ratio  = round(body, 3),
        dir_consist = round(direction, 3),
        tradeable   = tradeable,
        penalty     = round(penalty, 1),
    )
