# ============================================================
#  成交量 & 持仓量分析模块
#  这是期货独有的维度，与股票最大的区别在于持仓量
# ============================================================
"""
成交量状态（5种）
─────────────────
量增价涨  BULL_CONFIRM    ✅ 多头确认，趋势可延续
量缩价涨  BULL_WEAK       ⚠️  涨势动能不足，谨慎追多
量增价跌  BEAR_CONFIRM    ✅ 空头确认，趋势可延续
量缩价跌  BEAR_WEAK       ⚠️  跌势动能不足，接近底部
量平价稳  NEUTRAL         ⭕ 无明确信号，观望

持仓量四象限（期货特有）
──────────────────────────
价格上涨 + 持仓量增加 → 多头新增，趋势强（LONG_BUILD_UP）
价格上涨 + 持仓量减少 → 空头平仓推升，动能弱（SHORT_COVER）
价格下跌 + 持仓量增加 → 空头新增，趋势强（SHORT_BUILD_UP）
价格下跌 + 持仓量减少 → 多头止损离场，接近底部（LONG_LIQUIDATION）

联合信号强度（0-100）
────────────────────
三维同向（结构+量+持仓）→ 90+
两维同向               → 60-80
矛盾信号               → 20-40
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


# ─────────────────────────────────────────────
#  数据结构
# ─────────────────────────────────────────────

@dataclass
class VolumeState:
    code:        str    # BULL_CONFIRM / BULL_WEAK / BEAR_CONFIRM / BEAR_WEAK / NEUTRAL
    label:       str    # 中文标签
    signal:      str    # "bull" / "bear" / "neutral"
    strength:    int    # 0-100
    description: str

    # 原始数据
    current_vol:  float
    avg_vol:      float
    vol_ratio:    float     # 当日量 / 均量
    price_chg_pct: float    # 当日涨跌幅 %


@dataclass
class OIState:
    code:        str    # LONG_BUILD_UP / SHORT_COVER / SHORT_BUILD_UP / LONG_LIQUIDATION / NO_DATA
    label:       str
    signal:      str    # "bull" / "bear" / "neutral"
    strength:    int    # 0-100
    description: str

    current_oi:  float
    prev_oi:     float
    oi_chg_pct:  float  # 持仓量变化 %
    price_chg_pct: float


@dataclass
class VolOIResult:
    vol_state:   VolumeState
    oi_state:    OIState
    combined_signal: str    # "strong_bull" / "bull" / "neutral" / "bear" / "strong_bear"
    combined_score:  int    # 0-100（越高越看多）
    summary:     str        # 一句话综合判断


# ─────────────────────────────────────────────
#  成交量分析
# ─────────────────────────────────────────────

# 成交量比率阈值
VOL_HIGH_THRESH  = 1.5   # 放量：当日量 > 均量 1.5 倍
VOL_LOW_THRESH   = 0.7   # 缩量：当日量 < 均量 0.7 倍
VOL_PERIOD       = 10    # 均量计算周期（日）


def analyze_volume(df: pd.DataFrame) -> VolumeState:
    """
    分析成交量状态。
    df 需含 volume / close 列，至少 15 行。
    """
    if len(df) < VOL_PERIOD + 2:
        return VolumeState("NEUTRAL", "数据不足", "neutral", 50,
                           "数据不足，无法判断", 0, 0, 1.0, 0)

    close_today = float(df["close"].iloc[-1])
    close_prev  = float(df["close"].iloc[-2])
    vol_today   = float(df["volume"].iloc[-1])
    avg_vol     = float(df["volume"].rolling(VOL_PERIOD).mean().iloc[-1])

    price_chg_pct = (close_today - close_prev) / close_prev * 100 if close_prev else 0
    vol_ratio     = vol_today / avg_vol if avg_vol > 0 else 1.0

    price_up = price_chg_pct > 0.1    # 涨超 0.1% 视为上涨
    price_dn = price_chg_pct < -0.1   # 跌超 0.1% 视为下跌
    vol_high = vol_ratio >= VOL_HIGH_THRESH
    vol_low  = vol_ratio <= VOL_LOW_THRESH

    # 量的强度：超越均量越多，强度越高
    vol_strength = int(min(100, (vol_ratio - 1) * 50 + 50)) if vol_ratio > 1 else int(max(0, vol_ratio * 50))

    if price_up and vol_high:
        return VolumeState(
            code="BULL_CONFIRM", label="量增价涨", signal="bull",
            strength=min(100, 60 + int((vol_ratio - 1.5) * 20)),
            description=f"成交量是均量的{vol_ratio:.1f}倍，量价齐升，多头积极入场",
            current_vol=vol_today, avg_vol=avg_vol,
            vol_ratio=round(vol_ratio, 2), price_chg_pct=round(price_chg_pct, 2),
        )
    elif price_up and vol_low:
        return VolumeState(
            code="BULL_WEAK", label="量缩价涨", signal="neutral",
            strength=40,
            description=f"成交量萎缩至均量的{vol_ratio:.1f}倍，涨势缺乏量能支撑，谨慎追多",
            current_vol=vol_today, avg_vol=avg_vol,
            vol_ratio=round(vol_ratio, 2), price_chg_pct=round(price_chg_pct, 2),
        )
    elif price_dn and vol_high:
        return VolumeState(
            code="BEAR_CONFIRM", label="量增价跌", signal="bear",
            strength=min(100, 60 + int((vol_ratio - 1.5) * 20)),
            description=f"成交量是均量的{vol_ratio:.1f}倍，量价背离（量放量跌），空头主导",
            current_vol=vol_today, avg_vol=avg_vol,
            vol_ratio=round(vol_ratio, 2), price_chg_pct=round(price_chg_pct, 2),
        )
    elif price_dn and vol_low:
        return VolumeState(
            code="BEAR_WEAK", label="量缩价跌", signal="neutral",
            strength=40,
            description=f"成交量萎缩，下跌动能不足，可能接近阶段性底部",
            current_vol=vol_today, avg_vol=avg_vol,
            vol_ratio=round(vol_ratio, 2), price_chg_pct=round(price_chg_pct, 2),
        )
    else:
        return VolumeState(
            code="NEUTRAL", label="量能平稳", signal="neutral",
            strength=50,
            description=f"成交量接近均值（{vol_ratio:.1f}倍），价格变动幅度小，无明确信号",
            current_vol=vol_today, avg_vol=avg_vol,
            vol_ratio=round(vol_ratio, 2), price_chg_pct=round(price_chg_pct, 2),
        )


# ─────────────────────────────────────────────
#  持仓量分析（期货特有）
# ─────────────────────────────────────────────

OI_CHG_THRESH = 2.0     # 持仓量变化超过 2% 视为有效变化


def analyze_open_interest(df: pd.DataFrame) -> OIState:
    """
    分析持仓量状态（四象限模型）。
    df 需含 open_interest / close 列。
    若无持仓量数据，返回 NO_DATA 状态。
    """
    has_oi = "open_interest" in df.columns and df["open_interest"].notna().sum() > 10

    if not has_oi:
        return OIState(
            code="NO_DATA", label="无持仓量数据", signal="neutral", strength=50,
            description="当前数据源未提供持仓量，建议参考交易所持仓排名数据",
            current_oi=0, prev_oi=0, oi_chg_pct=0, price_chg_pct=0,
        )

    oi_today   = float(df["open_interest"].iloc[-1])
    oi_prev    = float(df["open_interest"].iloc[-2])
    close_today = float(df["close"].iloc[-1])
    close_prev  = float(df["close"].iloc[-2])

    oi_chg_pct    = (oi_today - oi_prev) / oi_prev * 100 if oi_prev > 0 else 0
    price_chg_pct = (close_today - close_prev) / close_prev * 100 if close_prev > 0 else 0

    price_up = price_chg_pct > 0.1
    price_dn = price_chg_pct < -0.1
    oi_up    = oi_chg_pct > OI_CHG_THRESH
    oi_dn    = oi_chg_pct < -OI_CHG_THRESH

    # 强度：持仓量变化越大，信号越强
    oi_strength = int(min(50, abs(oi_chg_pct) * 5))

    if price_up and oi_up:
        return OIState(
            code="LONG_BUILD_UP",
            label="多头增仓",
            signal="bull",
            strength=min(100, 65 + oi_strength),
            description=(
                f"价格上涨{price_chg_pct:+.1f}% 同时持仓量增加{oi_chg_pct:+.1f}%，"
                "新多头资金入场，趋势强，做多信号最强"
            ),
            current_oi=oi_today, prev_oi=oi_prev,
            oi_chg_pct=round(oi_chg_pct, 2), price_chg_pct=round(price_chg_pct, 2),
        )

    elif price_up and oi_dn:
        return OIState(
            code="SHORT_COVER",
            label="空头平仓",
            signal="neutral",
            strength=45,
            description=(
                f"价格上涨{price_chg_pct:+.1f}% 但持仓量减少{oi_chg_pct:+.1f}%，"
                "上涨由空头平仓推动（非新多买入），动能有限，不宜追多"
            ),
            current_oi=oi_today, prev_oi=oi_prev,
            oi_chg_pct=round(oi_chg_pct, 2), price_chg_pct=round(price_chg_pct, 2),
        )

    elif price_dn and oi_up:
        return OIState(
            code="SHORT_BUILD_UP",
            label="空头增仓",
            signal="bear",
            strength=min(100, 65 + oi_strength),
            description=(
                f"价格下跌{price_chg_pct:+.1f}% 同时持仓量增加{oi_chg_pct:+.1f}%，"
                "新空头资金入场，下跌趋势强，做空信号最强"
            ),
            current_oi=oi_today, prev_oi=oi_prev,
            oi_chg_pct=round(oi_chg_pct, 2), price_chg_pct=round(price_chg_pct, 2),
        )

    elif price_dn and oi_dn:
        return OIState(
            code="LONG_LIQUIDATION",
            label="多头平仓",
            signal="neutral",
            strength=40,
            description=(
                f"价格下跌{price_chg_pct:+.1f}% 且持仓量减少{oi_chg_pct:+.1f}%，"
                "多头止损离场，下跌动能减弱，阶段底部可能形成"
            ),
            current_oi=oi_today, prev_oi=oi_prev,
            oi_chg_pct=round(oi_chg_pct, 2), price_chg_pct=round(price_chg_pct, 2),
        )

    else:
        return OIState(
            code="NEUTRAL",
            label="持仓量平稳",
            signal="neutral",
            strength=50,
            description=f"持仓量变化{oi_chg_pct:+.1f}%，变动不显著，市场处于观望",
            current_oi=oi_today, prev_oi=oi_prev,
            oi_chg_pct=round(oi_chg_pct, 2), price_chg_pct=round(price_chg_pct, 2),
        )


# ─────────────────────────────────────────────
#  联合信号
# ─────────────────────────────────────────────

_SIGNAL_SCORE = {"bull": 70, "bear": 30, "neutral": 50}


def combine_vol_oi(vol: VolumeState, oi: OIState) -> VolOIResult:
    """
    将成交量和持仓量信号合并，给出联合判断。
    """
    v_score = _SIGNAL_SCORE.get(vol.signal, 50)
    o_score = _SIGNAL_SCORE.get(oi.signal, 50) if oi.code != "NO_DATA" else 50

    # 合并分数：有 OI 数据时各占 50%，无 OI 数据时量能占全部
    if oi.code == "NO_DATA":
        combined_score = int(v_score * 0.7 + vol.strength * 0.3)
    else:
        combined_score = int(v_score * 0.45 + o_score * 0.45 + (vol.strength + oi.strength) / 2 * 0.1)

    if combined_score >= 72:
        signal = "strong_bull"
    elif combined_score >= 58:
        signal = "bull"
    elif combined_score <= 28:
        signal = "strong_bear"
    elif combined_score <= 42:
        signal = "bear"
    else:
        signal = "neutral"

    # 生成综合描述
    parts = [f"【量能】{vol.label}——{vol.description}"]
    if oi.code != "NO_DATA":
        parts.append(f"【持仓】{oi.label}——{oi.description}")

    signal_zh = {
        "strong_bull": "强烈做多信号",
        "bull":        "偏多信号",
        "neutral":     "中性，观望",
        "bear":        "偏空信号",
        "strong_bear": "强烈做空信号",
    }.get(signal, "中性")

    summary = f"{signal_zh}（量价持仓综合得分 {combined_score}/100）。" + " | ".join(parts)

    return VolOIResult(
        vol_state       = vol,
        oi_state        = oi,
        combined_signal = signal,
        combined_score  = combined_score,
        summary         = summary,
    )


def analyze_vol_oi(df: pd.DataFrame) -> VolOIResult:
    """对外主接口：一次返回完整的量价持仓分析结果。"""
    vol = analyze_volume(df)
    oi  = analyze_open_interest(df)
    return combine_vol_oi(vol, oi)


# ─────────────────────────────────────────────
#  历史量能趋势（用于图表）
# ─────────────────────────────────────────────

def volume_trend_series(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    计算历史每日的量能状态序列，供可视化使用。
    返回含 date / vol_ratio / vol_label / color 的 DataFrame。
    """
    result = []
    avg_vol = df["volume"].rolling(period).mean()

    for i in range(period, len(df)):
        row   = df.iloc[i]
        prev  = df.iloc[i - 1]
        ratio = row["volume"] / avg_vol.iloc[i] if avg_vol.iloc[i] > 0 else 1.0
        pchg  = (row["close"] - prev["close"]) / prev["close"] * 100

        if ratio >= VOL_HIGH_THRESH:
            label = "量增价涨" if pchg > 0 else "量增价跌"
            color = "#26a69a" if pchg > 0 else "#ef5350"
        elif ratio <= VOL_LOW_THRESH:
            label = "量缩"
            color = "#888888"
        else:
            label = "量平"
            color = "#aaaaaa"

        result.append({
            "date":      row["date"],
            "volume":    row["volume"],
            "vol_ratio": round(ratio, 2),
            "vol_label": label,
            "color":     color,
            "price_chg": round(pchg, 2),
        })

    return pd.DataFrame(result)
