from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from config import ALL_SYMBOLS, CONTRACT_SPECS, RISK_PARAMS, SYMBOL_SECTOR
from indicators import add_all_indicators
from structure_analyzer import StructureState, analyze_structure
from volume_oi_analyzer import VolOIResult, analyze_vol_oi


@dataclass(frozen=True)
class TimeframeObservation:
    symbol: str
    timeframe: str
    available: bool
    bars: int
    close: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    price_state: str = "数据不足"
    volume_state: str = "数据不足"
    oi_state: str = "数据不足"
    time_state: str = "数据不足"
    bias: str = "neutral"
    continuation: int = 0
    key_high: float = 0.0
    key_low: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    description: str = "数据不足，暂不判断"
    structure: StructureState | None = None
    vol_oi: VolOIResult | None = None


@dataclass(frozen=True)
class BetPlan:
    action: str
    direction: str
    grade: str
    risk_pct: float
    entry_trigger: str
    invalidation: str
    stop_loss: float
    target_1r: float
    target_2r: float
    lots: int
    reason: str


@dataclass(frozen=True)
class ObjectiveMarketReport:
    symbol: str
    name: str
    sector: str
    observations: dict[str, TimeframeObservation]
    current_story: str
    uncertainty: str
    bet_plan: BetPlan
    checklist: list[str] = field(default_factory=list)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(out):
        return default
    return out


def _prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"行情字段缺失: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    out = out[out["volume"].fillna(0) >= 0]
    return add_all_indicators(out).dropna(subset=["ATR"]).reset_index(drop=True)


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    src = df.copy()
    src["date"] = pd.to_datetime(src["date"], errors="coerce")
    src = src.dropna(subset=["date"]).set_index("date").sort_index()
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "open_interest" in src.columns:
        agg["open_interest"] = "last"
    out = src.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    if len(out) < 20:
        return out
    return add_all_indicators(out).dropna(subset=["ATR"]).reset_index(drop=True)


def _has_intraday_dates(df: pd.DataFrame) -> bool:
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    return bool((dates.dt.hour != 0).any() or (dates.dt.minute != 0).any())


def _price_bias(structure: StructureState, close: float) -> str:
    if structure.sub_state == "BREAKOUT_UP":
        return "bull"
    if structure.sub_state == "BREAKOUT_DN":
        return "bear"
    if structure.trend == "UPTREND" and close >= structure.support:
        return "bull"
    if structure.trend == "DOWNTREND" and close <= structure.resistance:
        return "bear"
    return "neutral"


def _score_bias(biases: Iterable[str]) -> int:
    score = 0
    for bias in biases:
        if bias in ("bull", "strong_bull"):
            score += 1
        elif bias in ("bear", "strong_bear"):
            score -= 1
    return score


def _time_state(df: pd.DataFrame, atr_pct: float, structure: StructureState) -> str:
    if len(df) < 30:
        return "样本不足"
    latest_range = _safe_float(df["high"].iloc[-1] - df["low"].iloc[-1])
    avg_range = _safe_float((df["high"] - df["low"]).rolling(20).mean().iloc[-1], latest_range)
    atr_series = (df["ATR"] / df["close"] * 100).dropna()
    atr_rank = float((atr_series <= atr_pct).mean()) if not atr_series.empty else 0.5

    if structure.sub_state in ("BREAKOUT_UP", "BREAKOUT_DN") and latest_range >= avg_range * 1.2:
        return "扩张突破期"
    if atr_rank <= 0.3 and structure.trend == "RANGE":
        return "收敛蓄势期"
    if atr_rank >= 0.75 and structure.trend != "RANGE":
        return "趋势加速期"
    if structure.trend == "RANGE":
        return "平衡震荡期"
    return "趋势运行期"


def analyze_timeframe(symbol: str, df: pd.DataFrame, timeframe: str, min_bars: int = 30) -> TimeframeObservation:
    if df is None or len(df) < min_bars:
        return TimeframeObservation(symbol=symbol, timeframe=timeframe, available=False, bars=0 if df is None else len(df))

    work = df.copy()
    if "ATR" not in work.columns or work["ATR"].isna().all():
        work = add_all_indicators(work)
    work = work.dropna(subset=["ATR"]).reset_index(drop=True)
    if len(work) < min_bars:
        return TimeframeObservation(symbol=symbol, timeframe=timeframe, available=False, bars=len(work))

    row = work.iloc[-1]
    close = _safe_float(row["close"])
    atr = _safe_float(row["ATR"], close * 0.015)
    atr_pct = atr / close * 100 if close > 0 else 0.0
    structure = analyze_structure(work)
    vol_oi = analyze_vol_oi(work)
    bias = _price_bias(structure, close)
    force_score = _score_bias([bias, vol_oi.vol_state.signal, vol_oi.oi_state.signal])
    continuation = int(max(0, min(100, 50 + force_score * 16 + (vol_oi.combined_score - 50) * 0.35)))
    t_state = _time_state(work, atr_pct, structure)

    if structure.sub_state == "BREAKOUT_UP":
        price_state = "价格向上突破关键高点"
    elif structure.sub_state == "BREAKOUT_DN":
        price_state = "价格向下跌破关键低点"
    elif structure.trend == "UPTREND":
        price_state = "高低点抬升，价格处于上行结构"
    elif structure.trend == "DOWNTREND":
        price_state = "高低点下移，价格处于下行结构"
    else:
        price_state = "价格处于震荡平衡区"

    oi_state = vol_oi.oi_state.label
    if vol_oi.oi_state.code == "NO_DATA":
        oi_state = "无持仓量数据，持续性降级"

    description = (
        f"{timeframe}：{price_state}；{vol_oi.vol_state.label}；"
        f"{oi_state}；时间状态为{t_state}；持续性评分{continuation}/100。"
    )

    return TimeframeObservation(
        symbol=symbol,
        timeframe=timeframe,
        available=True,
        bars=len(work),
        close=round(close, 2),
        atr=round(atr, 2),
        atr_pct=round(atr_pct, 2),
        price_state=price_state,
        volume_state=vol_oi.vol_state.label,
        oi_state=oi_state,
        time_state=t_state,
        bias=bias,
        continuation=continuation,
        key_high=round(structure.recent_high, 2),
        key_low=round(structure.recent_low, 2),
        support=round(structure.support, 2),
        resistance=round(structure.resistance, 2),
        description=description,
        structure=structure,
        vol_oi=vol_oi,
    )


def _timeframes(df_daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {
        "周线": _resample_ohlcv(df_daily, "W-FRI"),
        "日线": df_daily,
    }
    if _has_intraday_dates(df_daily):
        frames["小时线"] = _resample_ohlcv(df_daily, "60min")
    return frames


def _direction_from_observations(obs: dict[str, TimeframeObservation]) -> str:
    weights = {"周线": 1, "日线": 2, "小时线": 2}
    score = 0
    for tf, item in obs.items():
        if not item.available:
            continue
        weight = weights.get(tf, 1)
        if item.bias == "bull":
            score += weight
        elif item.bias == "bear":
            score -= weight
    if score >= 2:
        return "多"
    if score <= -2:
        return "空"
    return "观望"


def _build_bet_plan(symbol: str, obs: dict[str, TimeframeObservation], capital: float) -> BetPlan:
    daily = obs.get("日线")
    weekly = obs.get("周线")
    hourly = obs.get("小时线")
    if daily is None or not daily.available or daily.structure is None:
        return BetPlan("WAIT", "观望", "禁止", 0, "等待有效日线数据", "数据不足", 0, 0, 0, 0, "数据不足")

    direction = _direction_from_observations(obs)
    conflicts = []
    if weekly and weekly.available and weekly.bias != "neutral" and daily.bias != "neutral" and weekly.bias != daily.bias:
        conflicts.append("周线与日线方向冲突")
    if hourly and hourly.available and hourly.bias != "neutral" and daily.bias != "neutral" and hourly.bias != daily.bias:
        conflicts.append("小时线与日线方向冲突")

    close = daily.close
    atr = daily.atr if daily.atr > 0 else close * 0.015
    lot_value = float(CONTRACT_SPECS.get(symbol, {}).get("lot_value", RISK_PARAMS["default_lot_value"]))
    confirm = daily.continuation
    has_oi = daily.vol_oi is not None and daily.vol_oi.oi_state.code != "NO_DATA"
    if not has_oi:
        confirm -= 8

    if direction == "多":
        stop = min(daily.support, daily.key_low) - 0.5 * atr
        if stop <= 0 or stop >= close:
            stop = close - 1.5 * atr
        risk = close - stop
        target_1r = close + risk
        target_2r = close + 2 * risk
        trigger = f"突破并站稳 {daily.resistance:.2f}，或回踩 {daily.support:.2f} 不破后放量转强"
        invalid = f"跌破 {stop:.2f} 或突破后重新收回区间"
    elif direction == "空":
        stop = max(daily.resistance, daily.key_high) + 0.5 * atr
        if stop <= close:
            stop = close + 1.5 * atr
        risk = stop - close
        target_1r = close - risk
        target_2r = close - 2 * risk
        trigger = f"跌破并收在 {daily.support:.2f} 下方，或反弹 {daily.resistance:.2f} 受阻后放量转弱"
        invalid = f"突破 {stop:.2f} 或跌破后重新收回区间"
    else:
        stop = 0.0
        risk = 0.0
        target_1r = 0.0
        target_2r = 0.0
        trigger = "等待价格离开震荡区，要求放量且持仓量支持方向"
        invalid = "方向未形成前不下注"

    if direction == "观望" or conflicts:
        action = "WAIT"
        grade = "观察"
        risk_pct = 0.0
        reason = "；".join(conflicts) if conflicts else "价格、成交量、持仓量未形成一致方向"
    elif confirm >= 72 and daily.time_state in ("扩张突破期", "趋势运行期", "趋势加速期"):
        action = "BET"
        grade = "A"
        risk_pct = 0.012
        reason = "方向与持续性一致，可等触发后下注"
    elif confirm >= 60:
        action = "SMALL_BET"
        grade = "B"
        risk_pct = 0.006
        reason = "方向初步一致，只允许小仓试错"
    else:
        action = "WAIT"
        grade = "观察"
        risk_pct = 0.0
        reason = "持续性不足，等待成交量和持仓量确认"

    risk_amount = capital * risk_pct
    risk_per_lot = risk * lot_value
    lots = int(risk_amount / risk_per_lot) if risk_amount > 0 and risk_per_lot > 0 else 0
    if action != "WAIT" and lots <= 0:
        action = "WAIT"
        grade = "观察"
        risk_pct = 0.0
        reason = "止损距离过大或合约乘数导致无法按风险开仓"

    return BetPlan(
        action=action,
        direction=direction,
        grade=grade,
        risk_pct=round(risk_pct, 4),
        entry_trigger=trigger,
        invalidation=invalid,
        stop_loss=round(stop, 2),
        target_1r=round(target_1r, 2),
        target_2r=round(target_2r, 2),
        lots=lots,
        reason=reason,
    )


def analyze_objective_market(symbol: str, df: pd.DataFrame, capital: float = RISK_PARAMS["capital"]) -> ObjectiveMarketReport:
    daily = _prepare_daily(df)
    observations = {
        name: analyze_timeframe(symbol, frame, name, min_bars=20 if name == "周线" else 30)
        for name, frame in _timeframes(daily).items()
    }
    if "小时线" not in observations:
        observations["小时线"] = TimeframeObservation(
            symbol=symbol,
            timeframe="小时线",
            available=False,
            bars=0,
            description="当前数据为日线级别，无小时线数据；日内首小时策略需接入分钟/小时行情后启用",
        )

    bet_plan = _build_bet_plan(symbol, observations, capital)
    available_desc = [item.description for item in observations.values() if item.available]
    current_story = " ".join(available_desc) if available_desc else "有效数据不足，不能描述当下行情。"

    uncertainty_parts = []
    daily_obs = observations.get("日线")
    if daily_obs and daily_obs.available:
        if daily_obs.vol_oi and daily_obs.vol_oi.oi_state.code == "NO_DATA":
            uncertainty_parts.append("缺少持仓量，持续性判断降级")
        if daily_obs.time_state in ("平衡震荡期", "收敛蓄势期"):
            uncertainty_parts.append("行情仍在等待方向选择")
    if observations["小时线"].available is False:
        uncertainty_parts.append("缺少小时线，日内入场点不能精确定位")
    uncertainty = "；".join(uncertainty_parts) if uncertainty_parts else "主要维度暂时一致，但仍按触发价和止损执行"

    checklist = [
        "价格必须突破/回踩关键位后再执行，不提前预测",
        "成交量必须放大或至少不能明显缩量",
        "持仓量增加时提高持续性评级，持仓量缺失时自动降级",
        "先确定止损，再按单笔风险计算手数",
        "触发后下一根K线执行，回测和实盘均避免前视偏差",
    ]

    return ObjectiveMarketReport(
        symbol=symbol,
        name=ALL_SYMBOLS.get(symbol, symbol),
        sector=SYMBOL_SECTOR.get(symbol, ""),
        observations=observations,
        current_story=current_story,
        uncertainty=uncertainty,
        bet_plan=bet_plan,
        checklist=checklist,
    )


def analyze_all_objective_markets(
    all_data: dict[str, pd.DataFrame],
    capital: float = RISK_PARAMS["capital"],
) -> list[ObjectiveMarketReport]:
    reports: list[ObjectiveMarketReport] = []
    for symbol, df in all_data.items():
        if df is None or df.empty:
            continue
        try:
            reports.append(analyze_objective_market(symbol, df, capital=capital))
        except Exception:
            continue
    grade_rank = {"A": 3, "B": 2, "观察": 1, "禁止": 0}
    return sorted(
        reports,
        key=lambda r: (grade_rank.get(r.bet_plan.grade, 0), r.observations.get("日线", TimeframeObservation(r.symbol, "日线", False, 0)).continuation),
        reverse=True,
    )


def reports_to_dataframe(reports: list[ObjectiveMarketReport]) -> pd.DataFrame:
    rows = []
    for report in reports:
        daily = report.observations.get("日线")
        weekly = report.observations.get("周线")
        hourly = report.observations.get("小时线")
        rows.append({
            "品种": f"{report.name}({report.symbol})",
            "板块": report.sector,
            "周线": weekly.price_state if weekly and weekly.available else "无",
            "日线": daily.price_state if daily and daily.available else "无",
            "小时线": hourly.price_state if hourly and hourly.available else "无",
            "成交量": daily.volume_state if daily and daily.available else "无",
            "持仓量": daily.oi_state if daily and daily.available else "无",
            "时间状态": daily.time_state if daily and daily.available else "无",
            "持续性": daily.continuation if daily and daily.available else 0,
            "动作": report.bet_plan.action,
            "方向": report.bet_plan.direction,
            "等级": report.bet_plan.grade,
            "风险比例": report.bet_plan.risk_pct,
            "建议手数": report.bet_plan.lots,
            "触发条件": report.bet_plan.entry_trigger,
            "止损": report.bet_plan.stop_loss,
            "1R目标": report.bet_plan.target_1r,
            "2R目标": report.bet_plan.target_2r,
            "判断": report.bet_plan.reason,
        })
    return pd.DataFrame(rows)
