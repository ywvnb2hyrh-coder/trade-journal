"""
config.py
Central configuration for the trade-journal package.
"""

from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = ROOT_DIR / "data"
OUTPUT_DIR  = ROOT_DIR / "output"
JOURNAL_CSV = ROOT_DIR / "journal_entries" / "journal.csv"

# ── Account ───────────────────────────────────────────────────────────────────
ACCOUNT_SIZE: float        = 50_000.0
RISK_FREE_RATE: float      = 0.0525
TRADING_DAYS_PER_YEAR: int = 252
MIN_TRADES_FOR_RATIOS: int = 50

# ── Topstep CSV column mapping ────────────────────────────────────────────────
COLUMN_MAP: dict[str, str] = {
    "Id":            "trade_id",
    "ContractName":  "symbol",
    "EnteredAt":     "entry_time",
    "ExitedAt":      "exit_time",
    "EntryPrice":    "entry_price",
    "ExitPrice":     "exit_price",
    "PnL":           "gross_pnl",
    "Fees":          "fees",
    "Commissions":   "commissions",
    "Size":          "size",
    "Type":          "direction",
    "TradeDay":      "trade_day",
    "TradeDuration": "duration_raw",
}

# ── Instrument tick metadata ──────────────────────────────────────────────────
TICK_METADATA: dict[str, dict[str, float]] = {
    "ES":  {"tick_size": 0.25,     "tick_value": 12.50},
    "MES": {"tick_size": 0.25,     "tick_value": 1.25},
    "NQ":  {"tick_size": 0.25,     "tick_value": 5.00},
    "MNQ": {"tick_size": 0.25,     "tick_value": 0.50},
    "CL":  {"tick_size": 0.01,     "tick_value": 10.00},
    "MCL": {"tick_size": 0.01,     "tick_value": 1.00},
    "GC":  {"tick_size": 0.10,     "tick_value": 10.00},
    "MGC": {"tick_size": 0.10,     "tick_value": 1.00},
    "6E":  {"tick_size": 0.00005,  "tick_value": 6.25},
    "ZN":  {"tick_size": 0.015625, "tick_value": 15.625},
}

# ── Duration buckets (minutes) ────────────────────────────────────────────────
DURATION_BINS: list[float] = [
    0, 0.25, 0.75, 1, 2, 5, 10, 30, 60, 120, 240, float("inf")
]
DURATION_LABELS: list[str] = [
    "<15s", "15-45s", "45s-1m", "1-2m", "2-5m",
    "5-10m", "10-30m", "30-60m", "1-2h", "2-4h", "4h+",
]

# ── Professional light color palette ─────────────────────────────────────────
# Matches institutional dashboard aesthetic: white background, sharp edges,
# muted grays, clean green/red for P&L only.
COLORS: dict[str, str] = {
    "bg":        "#ffffff",   # page background
    "panel":     "#ffffff",   # card background
    "border":    "#e2e8f0",   # card borders — light gray
    "border2":   "#cbd5e1",   # stronger border for header line
    "text":      "#0f172a",   # primary text — near black
    "text2":     "#475569",   # secondary text — slate
    "text3":     "#94a3b8",   # muted text — light slate
    "green":     "#16a34a",   # positive P&L
    "green_bg":  "#f0fdf4",   # green card background tint
    "red":       "#dc2626",   # negative P&L
    "red_bg":    "#fef2f2",   # red card background tint
    "blue":      "#2563eb",   # accent / rolling Sharpe line
    "blue_bg":   "#eff6ff",   # blue tint
    "grid":      "#f1f5f9",   # chart gridlines — very light
    "header_bg": "#f8fafc",   # topbar background
    "tag_bg":    "#f1f5f9",   # instrument tag background
}


def rgba(key: str, alpha: float = 1.0) -> str:
    """
    Convert a palette color to an ``rgba(r,g,b,a)`` string for Plotly fills.

    Parameters
    ----------
    key : str
        Key in :data:`COLORS`.
    alpha : float
        Opacity, 0.0–1.0.

    Returns
    -------
    str
        CSS rgba string, e.g. ``"rgba(22,163,74,0.15)"``.

    Examples
    --------
    >>> rgba("green", 0.15)
    'rgba(22,163,74,0.15)'
    """
    hex_color = COLORS[key].lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
