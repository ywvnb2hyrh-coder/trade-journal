"""
charts.py
Interactive Plotly chart builders for the trading dashboard.

Each function accepts a :class:`~trade_journal.analytics.TradeAnalytics`
instance and returns a ``plotly.graph_objects.Figure``.
Charts are fully interactive: hover tooltips, zoom, pan, download.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analytics import TradeAnalytics
from .config import COLORS as C, rgba

log = logging.getLogger(__name__)

# ── Shared layout defaults ────────────────────────────────────────────────────

_LAYOUT_BASE = dict(
    paper_bgcolor=C["panel"],
    plot_bgcolor=C["panel"],
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
              size=11, color=C["text"]),
    margin=dict(l=55, r=20, t=36, b=45),
    legend=dict(
        bgcolor=C["panel"],
        bordercolor=C["border"],
        borderwidth=1,
        font=dict(size=10, color=C["text2"]),
    ),
    xaxis=dict(
        gridcolor=C["grid"],
        gridwidth=1,
        linecolor=C["border"],
        linewidth=1,
        tickfont=dict(size=10, color=C["text3"]),
        title_font=dict(color=C["text3"]),
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor=C["grid"],
        gridwidth=1,
        linecolor=C["border"],
        linewidth=1,
        tickfont=dict(size=10, color=C["text3"]),
        title_font=dict(color=C["text3"]),
        showgrid=True,
        zeroline=False,
    ),
    hoverlabel=dict(
        bgcolor="white",
        bordercolor=C["border2"],
        font=dict(family="-apple-system, sans-serif", size=12, color=C["text"]),
    ),
)


def _base_layout(**overrides: object) -> dict[str, object]:
    """Merge base layout with chart-specific overrides."""
    layout: dict[str, object] = dict(_LAYOUT_BASE)
    layout.update(overrides)
    return layout


def _title(text: str) -> dict[str, object]:
    return dict(
        text=text,
        font=dict(color=C["text"], size=13, family="-apple-system, sans-serif"),
        x=0.0,
        xanchor="left",
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def equity_and_drawdown(ta: TradeAnalytics) -> go.Figure:
    """
    Two-panel figure: cumulative equity curve with fill (top)
    and drawdown % panel (bottom).
    """
    daily = ta.daily
    dates = daily["date"].astype(str)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Cumulative net P&L", "Drawdown %"],
    )

    fig.add_trace(go.Scatter(
        x=dates, y=daily["cum_pnl"].round(2),
        mode="lines",
        line=dict(color=C["blue"], width=2),
        fill="tozeroy",
        fillcolor=rgba("blue", 0.08),
        name="Equity",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Equity: $%{y:,.2f}"
            "<extra></extra>"
        ),
    ), row=1, col=1)

    # Drawdown shading between equity and peak
    fig.add_trace(go.Scatter(
        x=dates, y=daily["peak"].round(2),
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=daily["cum_pnl"].round(2),
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=rgba("red", 0.12),
        name="Drawdown region",
        hoverinfo="skip",
    ), row=1, col=1)

    # Drawdown % bars
    fig.add_trace(go.Bar(
        x=dates, y=daily["drawdown_pct"].round(3),
        marker_color=rgba("red", 0.7),
        marker_line_width=0,
        name="Drawdown %",
        hovertemplate="<b>%{x}</b><br>Drawdown: %{y:.3f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        **_base_layout(
            title=_title("Equity curve & drawdown"),
            height=420,
            showlegend=False,
        )
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)

    for ann in fig.layout.annotations:
        ann.font.color = C["text2"]
        ann.font.size  = 11        

    return fig


def daily_pnl_bar(ta: TradeAnalytics) -> go.Figure:
    """Bar chart of daily net P&L with value labels."""
    daily  = ta.daily
    dates  = daily["date"].astype(str)
    colors = [C["green"] if v >= 0 else C["red"] for v in daily["net_pnl"]]

    fig = go.Figure(go.Bar(
        x=dates,
        y=daily["net_pnl"].round(2),
        marker_color=colors,
        marker_line_width=0,
        text=daily["net_pnl"].apply(lambda v: f"${v:+,.0f}"),
        textposition="outside",
        textfont=dict(size=10, color=C["text2"]),
        customdata=np.stack([daily["trades"], daily["win_rate"] * 100], axis=1),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Net P&L: $%{y:,.2f}<br>"
            "Trades: %{customdata[0]}<br>"
            "Win rate: %{customdata[1]:.1f}%"
            "<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_color=C["border2"], line_width=1)
    fig.update_layout(**_base_layout(title=_title("Daily net P&L"), height=260))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def rolling_sharpe_line(ta: TradeAnalytics) -> go.Figure:
    """Rolling Sharpe ratio line. Window auto-adjusts to available data."""
    rs    = ta.rolling_sharpe()
    dates = ta.daily["date"].astype(str)

    if rs.empty or rs.isna().all():
        return go.Figure().update_layout(**_base_layout(
            title=_title("Rolling Sharpe — insufficient data"), height=220
        ))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=rs.clip(lower=0),
        fill="tozeroy", fillcolor=rgba("blue", 0.08),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=rs.clip(upper=0),
        fill="tozeroy", fillcolor=rgba("red", 0.10),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=rs,
        mode="lines+markers",
        line=dict(color=C["blue"], width=2),
        marker=dict(size=5, color=C["blue"], line=dict(color="white", width=1)),
        name="Rolling Sharpe",
        hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=1.0, line_dash="dash", line_color=C["green"],
                  line_width=1, annotation_text="1.0 target",
                  annotation_font_color=C["green"],
                  annotation_font_size=10)
    fig.add_hline(y=0, line_color=C["border2"], line_width=1)

    fig.update_layout(**_base_layout(
        title=_title("Rolling Sharpe ratio"), height=230, showlegend=False
    ))
    return fig


def pnl_distribution(ta: TradeAnalytics) -> go.Figure:
    """Overlapping histogram of winning vs losing trade P&L."""
    wins   = ta.df.loc[ta.df["win"],  "net_pnl"]
    losses = ta.df.loc[~ta.df["win"], "net_pnl"]
    mean   = float(ta.df["net_pnl"].mean())

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=wins, name="Wins",
        marker_color=rgba("green", 0.7),
        marker_line_color=C["green"],
        marker_line_width=0.5,
        opacity=0.85,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Histogram(
        x=losses, name="Losses",
        marker_color=rgba("red", 0.7),
        marker_line_color=C["red"],
        marker_line_width=0.5,
        opacity=0.85,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=C["border2"], line_width=1)
    fig.add_vline(x=mean, line_color=C["blue"], line_width=1.5,
                  line_dash="dash",
                  annotation_text=f"Mean ${mean:,.0f}",
                  annotation_font_color=C["blue"],
                  annotation_font_size=10,
                  annotation_position="top right")

    fig.update_layout(**_base_layout(
        title=_title("P&L distribution"),
        barmode="overlay", height=280,
    ))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return fig


def pnl_by_symbol(ta: TradeAnalytics) -> go.Figure:
    """Horizontal bar chart of net P&L by root symbol."""
    pbs    = ta.pnl_by_symbol()
    colors = [C["green"] if v >= 0 else C["red"] for v in pbs["net_pnl"]]

    fig = go.Figure(go.Bar(
        x=pbs["net_pnl"].round(2),
        y=pbs.index,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        customdata=np.stack([pbs["trades"], pbs["win_rate"]], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Net P&L: $%{x:,.2f}<br>"
            "Trades: %{customdata[0]}<br>"
            "Win rate: %{customdata[1]:.1f}%"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color=C["border2"], line_width=1)
    fig.update_layout(**_base_layout(
        title=_title("Net P&L by symbol"),
        height=max(200, len(pbs) * 55),
    ))
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")
    return fig


def long_short_breakdown(ta: TradeAnalytics) -> go.Figure:
    """Three-panel: total P&L, trade count, avg P&L by direction."""
    pbd = ta.pnl_by_direction().reset_index()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Total P&L", "Trade count", "Avg P&L"],
        horizontal_spacing=0.10,
    )

    for col_idx, (col, is_dollar) in enumerate(
        [("net_pnl", True), ("trades", False), ("avg_pnl", True)], start=1
    ):
        vals = pbd[col].values
        colors = (
            [C["green"] if v >= 0 else C["red"] for v in vals]
            if is_dollar else ["#6366f1"] * len(vals)
        )
        fig.add_trace(go.Bar(
            x=pbd["direction"], y=vals,
            marker_color=colors,
            marker_line_width=0,
            text=[f"${v:,.0f}" if is_dollar else str(int(v)) for v in vals],
            textposition="outside",
            textfont=dict(size=11, color=C["text2"]),
            showlegend=False,
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:,.2f}}<extra></extra>",
        ), row=1, col=col_idx)
        fig.add_hline(y=0, row=1, col=col_idx,
                      line_color=C["border2"], line_width=1)

    fig.update_layout(**_base_layout(title=_title("Long vs short breakdown"), height=270))
    for ann in fig.layout.annotations:
        ann.font.color = C["text2"] 
        ann.font.size  = 11         
    return fig


def duration_analysis(ta: TradeAnalytics) -> go.Figure:
    """Two-panel: win rate and net P&L by trade duration bucket."""
    pbd    = ta.pnl_by_duration()
    labels = pbd.index.astype(str)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Win rate by duration", "Net P&L by duration"],
        horizontal_spacing=0.10,
    )

    wr_colors  = [C["green"] if v >= 50 else C["red"] for v in pbd["win_rate"]]
    pnl_colors = [C["green"] if v >= 0  else C["red"] for v in pbd["net_pnl"]]

    fig.add_trace(go.Bar(
        x=pbd["win_rate"], y=labels, orientation="h",
        marker_color=wr_colors, marker_line_width=0,
        text=[f"{v:.0f}%  (n={n})" for v, n in zip(pbd["win_rate"], pbd["trades"])],
        textposition="outside", textfont=dict(size=9, color=C["text2"]),
        hovertemplate="<b>%{y}</b><br>Win rate: %{x:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=pbd["net_pnl"], y=labels, orientation="h",
        marker_color=pnl_colors, marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>Net P&L: $%{x:,.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.add_vline(x=50, row=1, col=1, line_color=C["border2"],
                  line_dash="dash", line_width=1)
    fig.add_vline(x=0,  row=1, col=2, line_color=C["border2"], line_width=1)

    fig.update_layout(**_base_layout(title=_title("Trade duration analysis"), height=290))
    fig.update_xaxes(ticksuffix="%", row=1, col=1)
    fig.update_xaxes(tickprefix="$", tickformat=",.0f", row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font.color = C["text2"]
        ann.font.size  = 11        
    return fig


def time_of_day(ta: TradeAnalytics) -> go.Figure:
    """Two-panel: win rate and net P&L by hour of day."""
    pbh = ta.pnl_by_hour().reset_index()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Win rate by hour (ET)", "Net P&L by hour (ET)"],
        horizontal_spacing=0.10,
    )

    wr_colors  = [C["green"] if v >= 50 else C["red"] for v in pbh["win_rate"]]
    pnl_colors = [C["green"] if v >= 0  else C["red"] for v in pbh["net_pnl"]]

    fig.add_trace(go.Bar(
        x=pbh["hour"].astype(str), y=pbh["win_rate"],
        marker_color=wr_colors, marker_line_width=0,
        hovertemplate="<b>Hour %{x}:00</b><br>Win rate: %{y:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=50, row=1, col=1, line_color=C["border2"],
                  line_dash="dash", line_width=1)

    fig.add_trace(go.Bar(
        x=pbh["hour"].astype(str), y=pbh["net_pnl"],
        marker_color=pnl_colors, marker_line_width=0,
        hovertemplate="<b>Hour %{x}:00</b><br>Net P&L: $%{y:,.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=0, row=1, col=2, line_color=C["border2"], line_width=1)

    fig.update_layout(**_base_layout(title=_title("Time of day analysis"), height=260))
    fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font.color = C["text2"]
        ann.font.size  = 11        
    return fig


def weekday_analysis(ta: TradeAnalytics) -> go.Figure:
    """Two-panel: total and avg P&L by day of week."""
    pbd  = ta.pnl_by_weekday().reset_index()
    days = pbd["weekday"].str[:3]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Total P&L by day", "Avg P&L by day"],
        horizontal_spacing=0.10,
    )

    for col_idx, col in enumerate(["net_pnl", "avg_pnl"], start=1):
        vals   = pbd[col].values
        colors = [C["green"] if v >= 0 else C["red"] for v in vals]
        fig.add_trace(go.Bar(
            x=days, y=vals,
            marker_color=colors, marker_line_width=0,
            hovertemplate=f"<b>%{{x}}</b><br>{col}: $%{{y:,.2f}}<extra></extra>",
            showlegend=False,
        ), row=1, col=col_idx)
        fig.add_hline(y=0, row=1, col=col_idx,
                      line_color=C["border2"], line_width=1)

    fig.update_layout(**_base_layout(title=_title("Day of week analysis"), height=260))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    for ann in fig.layout.annotations:
        ann.font.color = C["text2"]
        ann.font.size  = 11        
    return fig


def fee_drag(ta: TradeAnalytics) -> go.Figure:
    """Gross vs net cumulative P&L with fee drag area."""
    daily = ta.daily
    dates = daily["date"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=daily["cum_gross"].round(2),
        mode="lines", line=dict(color="#6366f1", width=2),
        name="Gross P&L",
        hovertemplate="<b>%{x}</b><br>Gross: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=daily["cum_pnl"].round(2),
        mode="lines", line=dict(color=C["blue"], width=2),
        fill="tonexty", fillcolor=rgba("red", 0.10),
        name="Net P&L",
        hovertemplate="<b>%{x}</b><br>Net: $%{y:,.2f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title=_title("Gross vs net P&L — fee drag"),
        height=250,
    ))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig
