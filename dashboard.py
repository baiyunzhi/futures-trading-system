# ============================================================
#  可视化仪表板（Plotly Dash）
#  启动后浏览器访问 http://127.0.0.1:8050
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


# ─────────────────────────────────────────────
#  图表生成函数
# ─────────────────────────────────────────────

def make_score_heatmap(rank_df: pd.DataFrame) -> go.Figure:
    """品种评分热力图（按板块分组）。"""
    if rank_df.empty:
        return go.Figure()

    df = rank_df.copy()
    df["板块"] = df["symbol"].map(SYMBOL_SECTOR)

    fig = px.bar(
        df,
        x="name", y="score",
        color="score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        hover_data=["symbol", "direction", "rsi", "adx"],
        labels={"score": "综合评分", "name": "品种"},
        title="品种综合评分排行",
    )
    fig.update_layout(
        height=350,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#e0e0e0",
        coloraxis_colorbar=dict(title="评分"),
        xaxis_tickangle=-30,
        margin=dict(t=50, b=80),
    )
    return fig


def make_kline_chart(symbol: str, df: pd.DataFrame) -> go.Figure:
    """K 线 + MA + MACD + ATR 多子图。"""
    df = df.copy().tail(120)   # 显示近 120 个交易日

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.04,
        subplot_titles=[
            f"{ALL_SYMBOLS.get(symbol, symbol)} K线 + 均线 + 布林带",
            "MACD",
            "RSI",
        ],
    )

    # ── K 线 ──
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        name="K线",
    ), row=1, col=1)

    # 均线
    colors_ma = {"MA5": "#FFD700", "MA10": "#FFA500", "MA20": "#00BFFF", "MA60": "#FF69B4"}
    for ma, color in colors_ma.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[ma],
                line=dict(color=color, width=1),
                name=ma,
            ), row=1, col=1)

    # 布林带
    if "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["date"], df["date"][::-1]]),
            y=pd.concat([df["BB_UPPER"], df["BB_LOWER"][::-1]]),
            fill="toself", fillcolor="rgba(128,128,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="布林带",
        ), row=1, col=1)

    # ── MACD ──
    if "HIST" in df.columns:
        colors = ["#ef5350" if v < 0 else "#26a69a" for v in df["HIST"]]
        fig.add_trace(go.Bar(
            x=df["date"], y=df["HIST"],
            marker_color=colors, name="MACD柱",
        ), row=2, col=1)
    if "DIF" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["DIF"],
            line=dict(color="#FFD700", width=1), name="DIF",
        ), row=2, col=1)
    if "DEA" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["DEA"],
            line=dict(color="#FF69B4", width=1), name="DEA",
        ), row=2, col=1)

    # ── RSI ──
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["RSI"],
            line=dict(color="#00BFFF", width=1.5), name="RSI",
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=600,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#e0e0e0",
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
        margin=dict(t=60, b=20),
    )
    fig.update_xaxes(gridcolor="#2a2a3e")
    fig.update_yaxes(gridcolor="#2a2a3e")
    return fig


def make_equity_curve(portfolio_result: dict) -> go.Figure:
    """组合回测净值曲线。"""
    fig = go.Figure()

    results = portfolio_result.get("results", {})
    for sym, res in results.items():
        if res.equity_curve.empty:
            continue
        name = ALL_SYMBOLS.get(sym, sym)
        # 归一化为净值
        curve = res.equity_curve / res.equity_curve.iloc[0]
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve.values,
            mode="lines", name=name,
            hovertemplate=f"{name}<br>净值: %{{y:.3f}}<extra></extra>",
        ))

    # 基准线
    fig.add_hline(y=1.0, line_dash="dash", line_color="#888", opacity=0.5)

    fig.update_layout(
        title="各品种回测净值曲线",
        height=400,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#e0e0e0",
        xaxis=dict(gridcolor="#2a2a3e"),
        yaxis=dict(gridcolor="#2a2a3e", title="净值"),
        legend=dict(orientation="h", y=1.05),
        margin=dict(t=60, b=30),
    )
    return fig


# ─────────────────────────────────────────────
#  Dash App
# ─────────────────────────────────────────────

def create_app(
    rank_df:          pd.DataFrame,
    analysis_results: list,
    all_data_ind:     dict[str, pd.DataFrame],   # 含指标的数据
    portfolio_result: dict,
) -> dash.Dash:
    """
    创建并返回 Dash App 实例。

    Parameters
    ----------
    rank_df           : variety_selector.rank_symbols() 结果
    analysis_results  : market_analyzer.analyze_all() 结果
    all_data_ind      : {symbol: df_with_indicators}
    portfolio_result  : backtester.backtest_portfolio() 结果
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],
        title="商品期货交易系统",
    )

    symbol_options = [
        {"label": f"{ALL_SYMBOLS.get(s, s)} ({s})", "value": s}
        for s in all_data_ind
    ]
    default_symbol = rank_df["symbol"].iloc[0] if not rank_df.empty else list(all_data_ind.keys())[0]

    # ── 市场状态表 ──
    from market_analyzer import results_to_dataframe
    state_df = results_to_dataframe(analysis_results)

    # ── 回测汇总表 ──
    summary_df = portfolio_result.get("summary", pd.DataFrame())

    # ── 样式 ──
    CARD_STYLE = {
        "backgroundColor": "#161b27",
        "border": "1px solid #2a2a3e",
        "borderRadius": "8px",
        "padding": "16px",
        "marginBottom": "16px",
    }

    def state_color(state: str) -> str:
        if "趋势做多" in state: return "#26a69a"
        if "趋势做空" in state: return "#ef5350"
        if "轻仓做多" in state: return "#FFD700"
        if "轻仓做空" in state: return "#FFA500"
        return "#888"

    # ── 顶部状态卡片 ──
    def make_state_cards():
        cards = []
        for r in analysis_results[:6]:
            color = state_color(r.state)
            name  = ALL_SYMBOLS.get(r.symbol, r.symbol)
            cards.append(dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{name}", style={"color": "#aaa", "margin": 0, "fontSize": "13px"}),
                        html.H4(r.state, style={"color": color, "margin": "4px 0", "fontSize": "14px"}),
                        html.Span(f"评分 {r.score:.0f} | ATR {r.atr:.1f}",
                                  style={"color": "#666", "fontSize": "11px"}),
                    ])
                ], style={"backgroundColor": "#1a2035", "border": f"1px solid {color}",
                          "borderRadius": "6px", "padding": "4px"}),
                width=2,
            ))
        return dbc.Row(cards, className="mb-3")

    # ── Layout ──
    app.layout = dbc.Container([
        # 标题栏
        dbc.Row([
            dbc.Col(html.H3(
                "🔭 商品期货交易系统 · 模拟行情验证",
                style={"color": "#e0e0e0", "padding": "16px 0 8px"},
            ))
        ]),

        # 状态卡片
        make_state_cards(),

        # 品种评分热力图
        dbc.Row([
            dbc.Col(
                dbc.Card([dcc.Graph(figure=make_score_heatmap(rank_df))],
                         style=CARD_STYLE),
            )
        ]),

        # K 线图（带品种选择）
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div([
                        html.Label("选择品种：", style={"color": "#aaa", "marginRight": "8px"}),
                        dcc.Dropdown(
                            id="symbol-dropdown",
                            options=symbol_options,
                            value=default_symbol,
                            clearable=False,
                            style={"width": "220px", "display": "inline-block",
                                   "backgroundColor": "#161b27", "color": "#333"},
                        ),
                    ], style={"padding": "8px 16px"}),
                    dcc.Graph(id="kline-chart"),
                ], style=CARD_STYLE),
            ])
        ]),

        # 回测净值曲线
        dbc.Row([
            dbc.Col(
                dbc.Card([dcc.Graph(figure=make_equity_curve(portfolio_result))],
                         style=CARD_STYLE),
            )
        ]),

        # 回测汇总 + 市场状态表
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.H5("📊 回测绩效汇总", style={"color": "#e0e0e0", "marginBottom": "12px"}),
                    dash_table.DataTable(
                        data=summary_df.to_dict("records") if not summary_df.empty else [],
                        columns=[{"name": c, "id": c} for c in summary_df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#161b27", "color": "#e0e0e0",
                                    "border": "1px solid #2a2a3e", "textAlign": "center",
                                    "fontSize": "12px", "padding": "6px"},
                        style_header={"backgroundColor": "#0e1117", "fontWeight": "bold",
                                      "color": "#aaa"},
                        style_data_conditional=[
                            {"if": {"column_id": "总收益%", "filter_query": "{总收益%} > 0"},
                             "color": "#26a69a"},
                            {"if": {"column_id": "总收益%", "filter_query": "{总收益%} < 0"},
                             "color": "#ef5350"},
                        ],
                        page_size=10,
                    ),
                ], style=CARD_STYLE),
            ], width=6),

            dbc.Col([
                dbc.Card([
                    html.H5("🎯 当前市场状态", style={"color": "#e0e0e0", "marginBottom": "12px"}),
                    dash_table.DataTable(
                        data=state_df.to_dict("records") if not state_df.empty else [],
                        columns=[{"name": c, "id": c} for c in state_df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"backgroundColor": "#161b27", "color": "#e0e0e0",
                                    "border": "1px solid #2a2a3e", "textAlign": "left",
                                    "fontSize": "11px", "padding": "5px 8px"},
                        style_header={"backgroundColor": "#0e1117", "fontWeight": "bold",
                                      "color": "#aaa"},
                        style_data_conditional=[
                            {"if": {"column_id": "市场状态",
                                    "filter_query": '{市场状态} contains "趋势做多"'},
                             "color": "#26a69a", "fontWeight": "bold"},
                            {"if": {"column_id": "市场状态",
                                    "filter_query": '{市场状态} contains "趋势做空"'},
                             "color": "#ef5350", "fontWeight": "bold"},
                            {"if": {"column_id": "市场状态",
                                    "filter_query": '{市场状态} contains "轻仓"'},
                             "color": "#FFD700"},
                        ],
                        page_size=10,
                    ),
                ], style=CARD_STYLE),
            ], width=6),
        ]),

        # 底部说明
        dbc.Row([
            dbc.Col(html.P(
                "⚠️ 本系统仅用于学习研究，不构成投资建议。期货交易存在较大风险，请谨慎操作。",
                style={"color": "#666", "fontSize": "12px", "textAlign": "center", "padding": "16px"},
            ))
        ]),

    ], fluid=True, style={"backgroundColor": "#0e1117", "minHeight": "100vh"})

    # ── Callback：更新 K 线 ──
    @app.callback(
        Output("kline-chart", "figure"),
        Input("symbol-dropdown", "value"),
    )
    def update_kline(symbol):
        if symbol and symbol in all_data_ind:
            return make_kline_chart(symbol, all_data_ind[symbol])
        return go.Figure()

    return app
