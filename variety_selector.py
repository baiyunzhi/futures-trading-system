# ============================================================
#  品种选择模块
#  对每个品种打分（0-100），筛选出当前最值得关注的标的
# ============================================================

import numpy as np
import pandas as pd
from config import SCORE_WEIGHTS, ALL_SYMBOLS, SYMBOL_SECTOR
from indicators import add_all_indicators, get_latest_row
from kline_density import analyze_density


# ─────────────────────────────────────────────
#  分项评分函数（各返回 0-100）
# ─────────────────────────────────────────────

def _trend_score(row: pd.Series, df: pd.DataFrame) -> float:
    """
    趋势强度评分。
    - MA5 > MA10 > MA20 > MA60：满分排列 +40
    - ADX 强度加分（最高 +40）
    - 价格在 MA20 上方/下方再加分
    """
    score = 0.0

    # MA 多头排列
    ma5, ma10, ma20, ma60 = row.get("MA5"), row.get("MA10"), row.get("MA20"), row.get("MA60")
    if all(pd.notna([ma5, ma10, ma20, ma60])):
        if ma5 > ma10:  score += 10
        if ma10 > ma20: score += 15
        if ma20 > ma60: score += 15

    # ADX 趋势强度（>25 算有趋势，>40 算强趋势）
    adx = row.get("ADX", np.nan)
    if pd.notna(adx):
        score += min(40, adx)   # ADX 直接映射到分数

    return min(100, score)


def _momentum_score(row: pd.Series, df: pd.DataFrame) -> float:
    """
    动量评分。
    - 近 5 日涨跌幅
    - 近 20 日涨跌幅
    - RSI 位置（50附近为中性，70以上空间压缩）
    """
    score = 50.0   # 中性基准

    if len(df) >= 20:
        ret5  = (df["close"].iloc[-1] / df["close"].iloc[-5]  - 1) * 100
        ret20 = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100
        score += np.clip(ret5  * 3, -20, 20)
        score += np.clip(ret20 * 1.5, -20, 20)

    rsi = row.get("RSI", np.nan)
    if pd.notna(rsi):
        # RSI 40-65 加分，<30 or >75 减分（极端）
        if 40 <= rsi <= 65:
            score += 10
        elif rsi < 30 or rsi > 75:
            score -= 10

    return np.clip(score, 0, 100)


def _volatility_score(row: pd.Series, df: pd.DataFrame) -> float:
    """
    波动率适度评分。
    过低（无趋势）或过高（风险大）均不理想，中间范围得分最高。
    """
    atr = row.get("ATR", np.nan)
    close = row.get("close", np.nan)
    if pd.isna(atr) or pd.isna(close) or close == 0:
        return 50.0

    atr_pct = atr / close * 100   # ATR 占价格比例（%）

    # 目标区间：0.8% - 2.5% 日波动为最佳
    if 0.8 <= atr_pct <= 2.5:
        score = 100 - abs(atr_pct - 1.6) * 20
    elif atr_pct < 0.8:
        score = atr_pct / 0.8 * 60   # 波动太低
    else:
        score = max(0, 100 - (atr_pct - 2.5) * 25)  # 波动太高

    # 量价配合：成交量 > 5日均量
    vol    = row.get("volume", np.nan)
    volma5 = row.get("VOL_MA5", np.nan)
    if pd.notna(vol) and pd.notna(volma5) and volma5 > 0:
        ratio = vol / volma5
        if ratio > 1.2:
            score = min(100, score + 10)

    return np.clip(score, 0, 100)


# ─────────────────────────────────────────────
#  综合评分
# ─────────────────────────────────────────────

def score_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """
    对单个品种打综合分，返回详情字典。
    密度评分作为惩罚项：K线越拥挤，总分扣分越多（最多扣35分）。
    """
    df_ind = add_all_indicators(df)
    row    = get_latest_row(df_ind)

    ts = _trend_score(row, df_ind)
    ms = _momentum_score(row, df_ind)
    vs = _volatility_score(row, df_ind)

    base_score = (
        ts * SCORE_WEIGHTS["trend"]
        + ms * SCORE_WEIGHTS["momentum"]
        + vs * SCORE_WEIGHTS["volatility"]
    )

    # ── K线密度惩罚 ──
    density = analyze_density(df_ind)
    total   = max(0.0, base_score - density.penalty)

    # 趋势方向：MA5 与 MA20 比较
    direction = "多" if row.get("MA5", 0) > row.get("MA20", 0) else "空"

    return {
        "symbol":        symbol,
        "name":          ALL_SYMBOLS.get(symbol, symbol),
        "sector":        SYMBOL_SECTOR.get(symbol, ""),
        "score":         round(total, 1),
        "base_score":    round(base_score, 1),
        "trend_score":   round(ts, 1),
        "mom_score":     round(ms, 1),
        "vol_score":     round(vs, 1),
        "density_score": density.score,
        "density_label": density.label,
        "density_color": density.color,
        "direction":     direction,
        "close":         round(float(row.get("close", 0)), 1),
        "atr":           round(float(row.get("ATR", 0)), 1),
        "rsi":           round(float(row.get("RSI", 50)), 1),
        "adx":           round(float(row.get("ADX", 0)), 1),
        "ma5":           round(float(row.get("MA5", 0)), 1),
        "ma20":          round(float(row.get("MA20", 0)), 1),
        "df_ind":        df_ind,
        "density":       density,   # 完整密度对象供后续使用
    }


def rank_symbols(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    对所有品种打分并排序，返回评分排行 DataFrame。
    """
    rows = []
    for symbol, df in all_data.items():
        if len(df) < 60:
            continue
        try:
            info = score_symbol(symbol, df)
            rows.append({k: v for k, v in info.items() if k != "df_ind"})
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"评分 {symbol} 异常: {e}")

    rank_df = pd.DataFrame(rows)
    if not rank_df.empty:
        rank_df = rank_df.sort_values("score", ascending=False).reset_index(drop=True)
        rank_df.index += 1
    return rank_df


def get_top_symbols(rank_df: pd.DataFrame, top_n: int = 5) -> list[str]:
    """返回评分最高的 Top N 品种代码列表。"""
    return rank_df["symbol"].head(top_n).tolist()
