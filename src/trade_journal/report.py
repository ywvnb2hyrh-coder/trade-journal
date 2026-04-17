"""
report.py
Assembles the final HTML dashboard from chart figures and metric data.

Separated from chart logic and CLI entry point so each module
has a single responsibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from .analytics import TradeAnalytics
from .charts import (
    daily_pnl_bar,
    duration_analysis,
    equity_and_drawdown,
    fee_drag,
    long_short_breakdown,
    pnl_by_symbol,
    pnl_distribution,
    rolling_sharpe_line,
    time_of_day,
    weekday_analysis,
)
from .config import COLORS as C
from .utils import fmt_dollar, fmt_pct, fmt_ratio

log = logging.getLogger(__name__)


def _fig_to_html(fig: go.Figure, div_id: str) -> str:
    """Convert a Plotly figure to an embeddable HTML div."""
    result: str = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {
                "format": "png", "filename": f"chart_{div_id}", "scale": 2
            },
        },
    )
    return result


def _metric_card(
    label: str,
    value: str,
    color: str = "",
    sub: str = "",
    bg: str = "",
) -> str:
    """
    Render one metric card.

    Parameters
    ----------
    label : str   Upper-case label above the value.
    value : str   Formatted metric value.
    color : str   CSS color for the value text.
    sub   : str   Optional small note below the value.
    bg    : str   Optional background color for the card.
    """
    color_style = f"color:{color};" if color else f"color:{C['text']};"
    bg_style    = f"background:{bg};" if bg else ""
    sub_html    = f'<div class="mc-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="mc" style="{bg_style}">'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value" style="{color_style}">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


def _panel(title: str, content_html: str) -> str:
    return (
        f'<div class="panel">'
        f'<div class="panel-title">{title}</div>'
        f'{content_html}'
        f'</div>'
    )


def _two_col(left_html: str, right_html: str) -> str:
    return (
        f'<div class="two-col">'
        f'<div class="col">{left_html}</div>'
        f'<div class="col">{right_html}</div>'
        f'</div>'
    )


def build(ta: TradeAnalytics, output_path: Path) -> Path:
    """
    Render all charts and write the complete HTML dashboard to disk.

    Parameters
    ----------
    ta : TradeAnalytics
        Loaded and cleaned analytics instance.
    output_path : Path
        Destination path for the HTML file.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    log.info("Building dashboard...")
    m    = ta.summary()
    mdd  = ta.max_drawdown()
    syms = "  ·  ".join(ta.pnl_by_symbol().index.tolist())
    now  = pd.Timestamp.now().strftime("%B %d, %Y  %H:%M")

    # ── Color helpers ─────────────────────────────────────────────────────
    def _pnl_color(v: float) -> str:
        return C["green"] if v >= 0 else C["red"]

    def _pnl_bg(v: float) -> str:
        return C["green_bg"] if v >= 0 else C["red_bg"]

    def _pf_color(v: float) -> str:
        return C["green"] if v >= 1.5 else C["red"]

    def _sh_color(v: float) -> str:
        return C["green"] if (not np.isnan(v) and v >= 1.0) else C["text2"]

    def _wr_color(v: float) -> str:
        return C["green"] if v >= 50 else C["red"]

    sortino_str = fmt_ratio(m["sortino_ratio"])
    calmar_str  = fmt_ratio(m["calmar_ratio"])
    rf_str      = fmt_ratio(m["recovery_factor"])

    # ── Metric cards — three visual groups ───────────────────────────────
    # Group 1: primary P&L (highlighted background)
    group1 = "".join([
        _metric_card("Realized P&L",    fmt_dollar(m["total_net_pnl"]),
                     _pnl_color(m["total_net_pnl"]), bg=_pnl_bg(m["total_net_pnl"])),
        _metric_card("Gross P&L",       fmt_dollar(m["total_gross_pnl"]),
                     C["text"]),
        _metric_card("Fees paid",       fmt_dollar(-m["total_fees"]),
                     C["red"], bg=C["red_bg"]),
        _metric_card("Win rate",        fmt_pct(m["win_rate_%"]),
                     _wr_color(m["win_rate_%"])),
        _metric_card("Profit factor",   fmt_ratio(m["profit_factor"]),
                     _pf_color(m["profit_factor"])),
        _metric_card("Expectancy",      fmt_dollar(m["expectancy"]),
                     _pnl_color(m["expectancy"])),
    ])

    # Group 2: trade statistics
    group2 = "".join([
        _metric_card("Avg win",         fmt_dollar(m["avg_win"]),    C["green"]),
        _metric_card("Avg loss",        fmt_dollar(m["avg_loss"]),   C["red"]),
        _metric_card("Win/loss ratio",  f"{fmt_ratio(m['win_loss_ratio'])}:1", C["text"]),
        _metric_card("Max drawdown",    fmt_pct(abs(m["max_drawdown_%"])),
                     C["red"], sub=fmt_dollar(mdd["$"])),
        _metric_card("Consec wins",     str(m["max_win_streak"]),    C["green"]),
        _metric_card("Consec losses",   str(m["max_loss_streak"]),   C["red"]),
    ])

    # Group 3: risk ratios
    group3 = "".join([
        _metric_card("Sharpe ratio",    fmt_ratio(m["sharpe_ratio"]),   _sh_color(m["sharpe_ratio"])),
        _metric_card("Sortino ratio",   sortino_str,                     C["text2"],
                     sub="needs losing days"),
        _metric_card("Calmar ratio",    calmar_str,                      C["text2"],
                     sub="needs 50+ trades"),
        _metric_card("Recovery factor", rf_str,                          C["text2"],
                     sub="needs 50+ trades"),
        _metric_card("Total trades",    str(m["total_trades"]),          C["text"]),
        _metric_card("Trading days",    str(m["trading_days"]),          C["text"]),
    ])

    # ── Build charts ──────────────────────────────────────────────────────
    log.info("Rendering charts...")
    charts = {
        "equity":   equity_and_drawdown(ta),
        "daily":    daily_pnl_bar(ta),
        "sharpe":   rolling_sharpe_line(ta),
        "dist":     pnl_distribution(ta),
        "symbol":   pnl_by_symbol(ta),
        "ls":       long_short_breakdown(ta),
        "duration": duration_analysis(ta),
        "hour":     time_of_day(ta),
        "weekday":  weekday_analysis(ta),
        "fee":      fee_drag(ta),
    }
    divs = {k: _fig_to_html(v, k) for k, v in charts.items()}

    # ── CSS ───────────────────────────────────────────────────────────────
    css = f"""
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
        background: #f8fafc;
        color: {C['text']};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
        font-size: 13px;
        line-height: 1.5;
        -webkit-font-smoothing: antialiased;
    }}

    .page {{ max-width: 1400px; margin: 0 auto; padding: 20px 24px 40px; }}

    /* ── Header ── */
    .header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        background: {C['panel']};
        border: 1px solid {C['border']};
        padding: 18px 22px;
        margin-bottom: 16px;
    }}
    .header-left {{ display: flex; flex-direction: column; gap: 6px; }}
    .logo {{
        font-size: 17px;
        font-weight: 700;
        color: {C['text']};
        letter-spacing: -0.02em;
    }}
    .logo span {{ color: {C['blue']}; }}
    .tag-row {{ display: flex; gap: 6px; align-items: center; }}
    .tag {{
        font-size: 11px;
        font-weight: 500;
        color: {C['text2']};
        background: {C['tag_bg']};
        border: 1px solid {C['border']};
        padding: 2px 8px;
        letter-spacing: 0.01em;
    }}
    .header-right {{
        font-size: 11px;
        color: {C['text3']};
        text-align: right;
        line-height: 1.8;
    }}
    .header-right strong {{
        color: {C['text2']};
        font-weight: 600;
    }}

    /* ── Section label ── */
    .section-label {{
        font-size: 10px;
        font-weight: 600;
        color: {C['text3']};
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 16px 0 8px;
    }}

    /* ── Metric grid ── */
    .metrics {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 8px;
        margin-bottom: 8px;
    }}
    .mc {{
        background: {C['panel']};
        border: 1px solid {C['border']};
        padding: 12px 14px;
    }}
    .mc-label {{
        font-size: 10px;
        font-weight: 500;
        color: {C['text3']};
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}
    .mc-value {{
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }}
    .mc-sub {{
        font-size: 10px;
        color: {C['text3']};
        margin-top: 4px;
    }}

    /* ── Chart panels ── */
    .panel {{
        background: {C['panel']};
        border: 1px solid {C['border']};
        margin-bottom: 10px;
        padding: 16px 18px;
    }}
    .panel-title {{
        font-size: 12px;
        font-weight: 600;
        color: {C['text']};
        border-bottom: 1px solid {C['border']};
        padding-bottom: 10px;
        margin-bottom: 12px;
        letter-spacing: -0.01em;
    }}
    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 10px;
    }}
    .col {{
        background: {C['panel']};
        border: 1px solid {C['border']};
        padding: 16px 18px;
    }}

    /* ── Footer ── */
    .footer {{
        font-size: 10px;
        color: {C['text3']};
        border-top: 1px solid {C['border']};
        margin-top: 20px;
        padding-top: 12px;
        text-align: center;
        letter-spacing: 0.02em;
    }}

    @media (max-width: 900px) {{
        .metrics {{ grid-template-columns: repeat(3, 1fr); }}
        .two-col  {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
        .metrics {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    """

    # ── Final HTML ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Trading Performance Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>{css}</style>
</head>
<body>
<div class="page">

  <div class="header">
    <div class="header-left">
      <div class="logo">Trading <span>Performance</span> Dashboard</div>
      <div class="tag-row">
        <span class="tag">TOPSTEP FUNDED</span>
        <span class="tag">INSTRUMENTS: {syms}</span>
        <span class="tag">ACCOUNT: ${ta.account_size:,.0f}</span>
      </div>
    </div>
    <div class="header-right">
      <strong>Generated</strong> {now}<br>
      Source: Topstep funded account export<br>
      All figures net of commissions and fees
    </div>
  </div>

  <div class="section-label">Performance summary</div>
  <div class="metrics">{group1}</div>

  <div class="section-label">Trade statistics</div>
  <div class="metrics">{group2}</div>

  <div class="section-label">Risk ratios</div>
  <div class="metrics">{group3}</div>

  {_panel("Equity curve & drawdown", divs["equity"])}
  {_panel("Daily net P&L", divs["daily"])}
  {_panel("Rolling Sharpe ratio", divs["sharpe"])}

  {_two_col(
      '<div class="panel-title">P&L distribution</div>' + divs["dist"],
      '<div class="panel-title">Net P&L by symbol</div>' + divs["symbol"],
  )}

  {_panel("Long vs short breakdown", divs["ls"])}
  {_panel("Trade duration analysis", divs["duration"])}
  {_panel("Time of day analysis", divs["hour"])}
  {_panel("Day of week analysis", divs["weekday"])}
  {_panel("Gross vs net P&L — fee drag", divs["fee"])}

  <div class="footer">
    Track record verified via Topstep funded account &nbsp;·&nbsp;
    All P&amp;L figures net of commissions and fees &nbsp;·&nbsp;
    Past performance does not guarantee future results
  </div>

</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    log.info("Dashboard written to %s", output_path.resolve())
    return output_path.resolve()
