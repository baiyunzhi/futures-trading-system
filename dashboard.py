# ============================================================
#  可视化仪表板 v3 — Plotly Dash
#  访问 http://127.0.0.1:8050
# ============================================================
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

from config import ALL_SYMBOLS, SYMBOL_SECTOR
from market_analyzer import AnalysisResult, get_detail_text, results_to_dataframe
from structure_analyzer import find_pivots
from kline_density import choppiness_index, density_score_series

CARD = {"backgroundColor": "#161b27", "border": "1px solid #2a2a3e",
        "borderRadius": "8px", "padding": "14px", "marginBottom": "14px"}
STATE_COLOR = {
    "trend_long":  "#26a69a", "trend_short": "#ef5350",
    "light_long":  "#FFD700", "light_short": "#FFA500", "observe": "#666",
}

def _state_color(state: str) -> str:
    if "趋势做多" in state: return "#26a69a"
    if "趋势做空" in state: return "#ef5350"
    if "轻仓做多" in state: return "#FFD700"
    if "轻仓做空" in state: return "#FFA500"
    return "#666666"


# ─────────────────────────────────────────────
#  K 线图表（含高低点标注 + 密度子图）
# ─────────────────────────────────────────────

def build_kline_chart(df: pd.DataFrame, symbol: str, result: AnalysisResult) -> go.Figure:
    """5行子图：K线+MA+枢轴 / MACD / 成交量 / CI密度 / 持仓量（若有）"""
    has_oi = "open_interest" in df.columns and df["open_interest"].notna().any()
    rows = 5 if has_oi else 4
    row_heights = [0.45, 0.15, 0.15, 0.15, 0.10] if has_oi else [0.45, 0.18, 0.18, 0.19]
    subplot_titles = ["K线 + 均线 + 枢轴", "MACD", "成交量", "震荡指数(CI) + 密度", "持仓量"] if has_oi else ["K线 + 均线 + 枢轴", "MACD", "成交量", "震荡指数(CI) + 密度"]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=row_heights,
                        subplot_titles=subplot_titles)

    tail = df.tail(120).copy()
    x = tail["date"].astype(str)

    # ── Row 1: K线 ──
    fig.add_trace(go.Candlestick(x=x, open=tail["open"], high=tail["high"],
                                  low=tail["low"], close=tail["close"],
                                  increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                                  name="K线", showlegend=False), row=1, col=1)

    for col, color, w in [("MA5","#FFD700",1),("MA10","#FF8C00",1),("MA20","#00BFFF",1.5),("MA60","#FF69B4",2)]:
        if col in tail.columns:
            fig.add_trace(go.Scatter(x=x, y=tail[col], line=dict(color=color, width=w),
                                     name=col, showlegend=True), row=1, col=1)

    # 枢轴标注  — find_pivots 返回 (highs, lows) PivotPoint 列表
    try:
        ph, pl = find_pivots(df.tail(200))
        # 过滤出在 tail(120) 日期范围内的枢轴
        tail_dates = set(tail["date"].astype(str))
        if ph:
            ph_filtered = [p for p in ph if str(p.date.date()) in tail_dates]
            if ph_filtered:
                fig.add_trace(go.Scatter(
                    x=[str(p.date.date()) for p in ph_filtered],
                    y=[p.price for p in ph_filtered],
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
                    text=["H"]*len(ph_filtered), textposition="top center",
                    name="枢轴高点", showlegend=True), row=1, col=1)
        if pl:
            pl_filtered = [p for p in pl if str(p.date.date()) in tail_dates]
            if pl_filtered:
                fig.add_trace(go.Scatter(
                    x=[str(p.date.date()) for p in pl_filtered],
                    y=[p.price for p in pl_filtered],
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
                    text=["L"]*len(pl_filtered), textposition="bottom center",
                    name="枢轴低点", showlegend=True), row=1, col=1)
    except Exception:
        pass

    # ── Row 2: MACD ──
    if "DIF" in tail.columns and "DEA" in tail.columns:
        hist = tail["DIF"] - tail["DEA"]
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
        fig.add_trace(go.Bar(x=x, y=hist, marker_color=colors, name="MACD柱", showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail["DIF"], line=dict(color="#FFD700", width=1), name="DIF"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=tail["DEA"], line=dict(color="#FF8C00", width=1), name="DEA"), row=2, col=1)

    # ── Row 3: 成交量（按涨跌 + 放量高亮着色）──
    if "volume" in tail.columns:
        vol_ma = tail["volume"].rolling(5).mean()
        bar_colors = []
        for i, (idx, row_s) in enumerate(tail.iterrows()):
            base = "#26a69a" if row_s["close"] >= row_s["open"] else "#ef5350"
            if pd.notna(vol_ma.iloc[i]) and row_s["volume"] > vol_ma.iloc[i] * 1.5:
                base = "#FFD700"  # 放量高亮
            bar_colors.append(base)
        fig.add_trace(go.Bar(x=x, y=tail["volume"], marker_color=bar_colors, name="成交量", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=vol_ma, line=dict(color="#aaa", width=1), name="量5MA", showlegend=False), row=3, col=1)

    # ── Row 4: CI + 密度评分 ──
    ci_series = choppiness_index(tail)
    den_df = density_score_series(tail)
    fig.add_trace(go.Scatter(x=x, y=ci_series.values, line=dict(color="#00BFFF", width=1.5),
                             name="CI(14)", showlegend=True), row=4, col=1)
    if "density_score" in den_df.columns:
        fig.add_trace(go.Scatter(x=x, y=den_df["density_score"].values,
                                 line=dict(color="#FFA500", width=1.5, dash="dot"),
                                 name="密度评分", showlegend=True), row=4, col=1)
    for val, color, dash in [(38.2, "#26a69a", "dash"), (61.8, "#ef5350", "dash")]:
        fig.add_hline(y=val, line_dash=dash, line_color=color, opacity=0.6, row=4, col=1)

    # ── Row 5: 持仓量 ──
    if has_oi:
        fig.add_trace(go.Bar(x=x, y=tail["open_interest"], marker_color="#9370DB",
                             name="持仓量", showlegend=False), row=5, col=1)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=720, margin=dict(l=40, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis_rangeslider_visible=False,
        title=dict(text=f"{result.name}（{symbol}）— {result.state}", font=dict(size=14)),
    )
    for i in range(1, rows + 1):
        fig.update_yaxes(showgrid=True, gridcolor="#2a2a3e", gridwidth=1, row=i, col=1)
        fig.update_xaxes(showgrid=False, row=i, col=1)
    return fig


# ─────────────────────────────────────────────
#  评分热力图 & 权益曲线
# ─────────────────────────────────────────────

def build_score_heatmap(rank_df: pd.DataFrame) -> go.Figure:
    """品种综合评分热力图"""
    if rank_df.empty:
        return go.Figure()
    cols = ["品种", "综合评分", "趋势", "动量", "波动率"]
    show_cols = [c for c in ["name", "score", "trend_score", "mom_score", "vol_score"] if c in rank_df.columns]
    display_df = rank_df[show_cols].head(15)
    labels = display_df.get("name", display_df.iloc[:,0]).tolist()
    vals   = display_df.get("score", display_df.iloc[:,1]).tolist()
    colors = [_state_color("") if v < 45 else "#FFD700" if v < 65 else "#26a69a" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                           marker_color=colors, text=[f"{v:.1f}" for v in vals],
                           textposition="inside"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      height=max(250, len(labels)*28), margin=dict(l=10, r=20, t=20, b=20),
                      xaxis=dict(range=[0, 105]), yaxis=dict(autorange="reversed"))
    return fig


def build_equity_chart(bt_results: dict) -> go.Figure:
    """组合权益曲线"""
    fig = go.Figure()
    if not bt_results:
        fig.add_annotation(text="暂无回测数据", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", height=250)
        return fig

    palette = px.colors.qualitative.Plotly
    for i, (sym, res) in enumerate(bt_results.items()):
        # res is a BacktestResult dataclass; .equity_curve is a pd.Series
        if hasattr(res, "equity_curve"):
            eq_s = res.equity_curve
        elif isinstance(res, dict):
            eq_s = res.get("equity_curve", pd.Series(dtype=float))
        else:
            continue
        if eq_s is None or len(eq_s) == 0:
            continue
        eq_s = pd.Series(eq_s.values if hasattr(eq_s, "values") else list(eq_s))
        fig.add_trace(go.Scatter(x=list(range(len(eq_s))), y=eq_s, name=sym,
                                 line=dict(color=palette[i % len(palette)], width=1.5)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      height=250, margin=dict(l=40, r=20, t=20, b=20),
                      legend=dict(font=dict(size=9), orientation="h", y=-0.2))
    return fig


# ─────────────────────────────────────────────
#  Dash App 组装
# ─────────────────────────────────────────────

def create_app(
    all_results: list[AnalysisResult],
    rank_df:     pd.DataFrame,
    bt_results:  dict,
    all_data:    dict[str, pd.DataFrame],
) -> dash.Dash:

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                    title="商品期货交易系统")

    # ── 结果 DataFrame ──
    result_df = results_to_dataframe(all_results)
    sym_map   = {r.symbol: r for r in all_results}
    symbols   = [r.symbol for r in all_results]
    first_sym = symbols[0] if symbols else ""

    # ── 状态徽章 ──
    def _badge(state):
        color = _state_color(state)
        return html.Span(state, style={"background": color, "color": "#000" if "黄" in state else "#fff",
                                       "padding": "2px 8px", "borderRadius": "4px", "fontSize": "12px"})

    # ── 表格列定义 ──
    TABLE_COLS = [
        {"name": c, "id": c,
         "type": "numeric" if c in ("综合评分","参考入场","止损","止盈","ATR") else "text"}
        for c in result_df.columns
    ] if not result_df.empty else []

    TABLE_COND = [
        {"if": {"filter_query": "{市场状态} contains '趋势做多'", "column_id": "市场状态"}, "color": "#26a69a"},
        {"if": {"filter_query": "{市场状态} contains '趋势做空'", "column_id": "市场状态"}, "color": "#ef5350"},
        {"if": {"filter_query": "{市场状态} contains '轻仓做多'", "column_id": "市场状态"}, "color": "#FFD700"},
        {"if": {"filter_query": "{市场状态} contains '轻仓做空'", "column_id": "市场状态"}, "color": "#FFA500"},
        {"if": {"filter_query": "{市场状态} contains '观望'",   "column_id": "市场状态"}, "color": "#888"},
        {"if": {"row_index": "odd"}, "backgroundColor": "#1a1f2e"},
    ]

    # ── Layout ──
    app.layout = dbc.Container([
        # ── 顶部标题栏 ──
        dbc.Row([
            dbc.Col(html.H4("商品期货交易系统", style={"color": "#26a69a", "margin": "0"}), width="auto"),
            dbc.Col(html.Small("实时分析 · K线密度 · 四维评分", style={"color": "#888", "lineHeight": "40px"}), width="auto"),
        ], align="center", className="mb-3 mt-2"),

        # ── 状态卡片 ──
        dbc.Row([
            dbc.Col(dbc.Card([
                html.Div("趋势做多", style={"color": "#26a69a", "fontWeight": "bold"}),
                html.H4(str(sum(1 for r in all_results if "趋势做多" in r.state)), style={"color": "#26a69a", "margin": "0"}),
            ], style={**CARD, "textAlign": "center"}), width=2),
            dbc.Col(dbc.Card([
                html.Div("趋势做空", style={"color": "#ef5350", "fontWeight": "bold"}),
                html.H4(str(sum(1 for r in all_results if "趋势做空" in r.state)), style={"color": "#ef5350", "margin": "0"}),
            ], style={**CARD, "textAlign": "center"}), width=2),
            dbc.Col(dbc.Card([
                html.Div("轻仓做多", style={"color": "#FFD700", "fontWeight": "bold"}),
                html.H4(str(sum(1 for r in all_results if "轻仓做多" in r.state)), style={"color": "#FFD700", "margin": "0"}),
            ], style={**CARD, "textAlign": "center"}), width=2),
            dbc.Col(dbc.Card([
                html.Div("轻仓做空", style={"color": "#FFA500", "fontWeight": "bold"}),
                html.H4(str(sum(1 for r in all_results if "轻仓做空" in r.state)), style={"color": "#FFA500", "margin": "0"}),
            ], style={**CARD, "textAlign": "center"}), width=2),
            dbc.Col(dbc.Card([
                html.Div("观望", style={"color": "#888", "fontWeight": "bold"}),
                html.H4(str(sum(1 for r in all_results if "观望" in r.state)), style={"color": "#888", "margin": "0"}),
            ], style={**CARD, "textAlign": "center"}), width=2),
            dbc.Col(dbc.Card([
                html.Div("品种总数", style={"color": "#aaa", "fontWeight": "bold"}),
                html.H4(str(len(all_results)), style={"color": "#aaa", "margin": "0"}),
            ], style={**CARD, "textAlign": "center"}), width=2),
        ], className="mb-3"),

        # ── K线图 + 详情 ──
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.Row([
                        dbc.Col(html.Label("选择品种：", style={"color": "#aaa"}), width="auto"),
                        dbc.Col(dcc.Dropdown(
                            id="symbol-selector",
                            options=[{"label": f"{r.name}({r.symbol}) [{r.state.split()[-1]}]", "value": r.symbol}
                                     for r in all_results],
                            value=first_sym,
                            style={"backgroundColor": "#1a1f2e", "color": "#eee", "border": "1px solid #333"},
                            className="mb-2",
                        ), width=8),
                    ], align="center"),
                    dcc.Graph(id="kline-chart", config={"scrollZoom": True, "displayModeBar": True}),
                ], style=CARD),
            ], width=8),
            dbc.Col([
                dbc.Card([
                    html.H6("行情深度分析", style={"color": "#26a69a", "marginBottom": "8px"}),
                    html.Pre(id="detail-text",
                             style={"color": "#ccc", "fontSize": "12px", "whiteSpace": "pre-wrap",
                                    "wordBreak": "break-word", "maxHeight": "650px", "overflowY": "auto",
                                    "backgroundColor": "#0e1117", "padding": "10px", "borderRadius": "6px"}),
                ], style=CARD),
            ], width=4),
        ], className="mb-3"),

        # ── 评分热力图 + 权益曲线 ──
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H6("品种综合评分排行", style={"color": "#26a69a", "marginBottom": "6px"}),
                    dcc.Graph(figure=build_score_heatmap(rank_df),
                              config={"displayModeBar": False}),
                ], style=CARD),
            ], width=5),
            dbc.Col([
                dbc.Card([
                    html.H6("历史权益曲线", style={"color": "#26a69a", "marginBottom": "6px"}),
                    dcc.Graph(figure=build_equity_chart(bt_results),
                              config={"displayModeBar": False}),
                ], style=CARD),
            ], width=7),
        ], className="mb-3"),

        # ── 市场状态总览表 ──
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H6("市场状态一览", style={"color": "#26a69a", "marginBottom": "8px"}),
                    dash_table.DataTable(
                        id="state-table",
                        columns=TABLE_COLS,
                        data=result_df.to_dict("records") if not result_df.empty else [],
                        sort_action="native",
                        filter_action="native",
                        page_size=12,
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#0e1117", "color": "#ccc",
                                    "border": "1px solid #2a2a3e", "fontSize": "12px",
                                    "padding": "4px 8px", "textAlign": "center"},
                        style_header={"backgroundColor": "#161b27", "fontWeight": "bold",
                                      "color": "#aaa", "border": "1px solid #333"},
                        style_data_conditional=TABLE_COND,
                    ),
                ], style=CARD),
            ], width=12),
        ], className="mb-3"),

        # ── 底部说明 ──
        dbc.Row([
            dbc.Col(html.P(
                "免责声明：本系统仅供学习研究，不构成投资建议。期货交易有风险，请谨慎操作。",
                style={"color": "#555", "fontSize": "11px", "textAlign": "center"},
            )),
        ]),

    ], fluid=True, style={"backgroundColor": "#0e1117", "minHeight": "100vh", "padding": "10px 20px"})


    # ─────────────────────────────────────────────
    #  回调：切换品种 → 更新K线图 + 详情文字
    # ─────────────────────────────────────────────

    @app.callback(
        Output("kline-chart", "figure"),
        Output("detail-text", "children"),
        Input("symbol-selector", "value"),
    )
    def update_chart(symbol: str):
        if not symbol or symbol not in sym_map:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                                plot_bgcolor="#0e1117", height=720)
            return empty, "请选择品种"

        result = sym_map[symbol]
        df     = all_data.get(symbol)
        if df is None or df.empty:
            empty = go.Figure()
            empty.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                                plot_bgcolor="#0e1117", height=720)
            return empty, f"{symbol} 暂无行情数据"

        fig  = build_kline_chart(df, symbol, result)
        text = get_detail_text(result)
        return fig, text

    return app
