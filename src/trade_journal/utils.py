"""
utils.py
Shared parsing, formatting, and validation utilities.

All functions are pure (no side effects) and fully vectorized where possible.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class TradeDataError(ValueError):
    """Raised when trade CSV data is malformed or missing required columns."""


class InsufficientDataError(ValueError):
    """Raised when there are too few observations to compute a metric."""


# ── Symbol helpers ────────────────────────────────────────────────────────────

_CONTRACT_RE = re.compile(r"^([A-Z0-9]+)[FGHJKMNQUVXZ]\d{1,2}$")


def root_symbol(symbol: str) -> str:
    """
    Strip futures contract month/year suffix to return the root symbol.

    Parameters
    ----------
    symbol : str
        Raw contract name, e.g. ``"MCLK6"``, ``"/ESH25"``, ``"MNQM6"``.

    Returns
    -------
    str
        Root symbol, e.g. ``"MCL"``, ``"ES"``, ``"MNQ"``.

    Examples
    --------
    >>> root_symbol("MCLK6")
    'MCL'
    >>> root_symbol("/ESH25")
    'ES'
    >>> root_symbol("MNQM6")
    'MNQ'
    >>> root_symbol("")
    'UNKNOWN'
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return "UNKNOWN"
    clean = symbol.upper().strip().lstrip("/")
    match = _CONTRACT_RE.match(clean)
    return match.group(1) if match else clean


def root_symbol_series(series: pd.Series) -> pd.Series:
    """
    Vectorized :func:`root_symbol` applied to a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Series of raw contract name strings.

    Returns
    -------
    pd.Series
        Root symbols as strings.
    """
    cleaned = (
        series.fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.lstrip("/")
    )
    def _extract(s: str) -> str:
        if not s:
            return "UNKNOWN"
        m = _CONTRACT_RE.match(s)
        return m.group(1) if m is not None else s

    return cleaned.apply(_extract)


# ── Duration parsing ──────────────────────────────────────────────────────────

def parse_duration_seconds(duration: object) -> float:
    """
    Parse Topstep's ``HH:MM:SS.ffffff`` duration string to total seconds.

    Parameters
    ----------
    duration : object
        Duration string from Topstep export, e.g. ``"00:11:09.8021060"``.
        Handles ``None``, ``float("nan")``, and empty strings gracefully.

    Returns
    -------
    float
        Total duration in seconds. Returns ``0.0`` for null/empty/unparseable input.

    Examples
    --------
    >>> parse_duration_seconds("00:11:09.8021060")
    669.8021060
    >>> parse_duration_seconds("01:00:00")
    3600.0
    >>> parse_duration_seconds(None)
    0.0
    >>> parse_duration_seconds("")
    0.0
    """
    if duration is None:
        return 0.0
    # pd.isna accepts object but mypy's stubs are strict — convert via str check first
    try:
        scalar = float(str(duration)) if isinstance(duration, (int, float)) else None
        if scalar is not None and (scalar != scalar):  # NaN check without pd.isna
            return 0.0
    except (ValueError, TypeError):
        pass
    # Also check via pandas for Series elements
    try:
        if isinstance(duration, float) and pd.isna(duration):
            return 0.0
    except (TypeError, ValueError):
        pass
    s = str(duration).strip()
    if not s:
        return 0.0
    try:
        parts   = s.split(":")
        hours   = float(parts[0]) if len(parts) == 3 else 0.0
        minutes = float(parts[1]) if len(parts) >= 2 else 0.0
        seconds = float(parts[-1])
        return hours * 3600.0 + minutes * 60.0 + seconds
    except (ValueError, IndexError):
        log.warning("Could not parse duration: %r — defaulting to 0", duration)
        return 0.0


def parse_duration_series(series: pd.Series) -> pd.Series:
    """
    Vectorized duration parser returning total minutes as float64.

    Parameters
    ----------
    series : pd.Series
        Series of Topstep duration strings.

    Returns
    -------
    pd.Series
        Duration in minutes (float64).
    """
    return (series.map(parse_duration_seconds) / 60.0).round(4)


# ── Safe metric wrapper ───────────────────────────────────────────────────────

def safe_metric(fn: object, *args: object, **kwargs: object) -> float:
    """
    Call a metric function and return ``nan`` on :exc:`InsufficientDataError`.

    Used in :meth:`~trade_journal.analytics.TradeAnalytics.summary` to
    gracefully handle metrics that require more data than is currently available.

    Parameters
    ----------
    fn : callable
        Metric function to call.
    *args
        Positional arguments forwarded to ``fn``.
    **kwargs
        Keyword arguments forwarded to ``fn``.

    Returns
    -------
    float
        Result of ``fn(*args, **kwargs)``, or ``nan`` if
        :exc:`InsufficientDataError` is raised.

    Examples
    --------
    >>> def always_raises():
    ...     raise InsufficientDataError("not enough days")
    >>> safe_metric(always_raises)
    nan
    """
    try:
        result = fn(*args, **kwargs)  # type: ignore[operator]
        return float(result)
    except InsufficientDataError:
        return float("nan")


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_dollar(value: float, sign: bool = True) -> str:
    """
    Format a float as a dollar string.

    Parameters
    ----------
    value : float
        Monetary value.
    sign : bool
        If ``True``, prefix negative values with ``-$`` instead of ``$-``.

    Returns
    -------
    str
        Formatted string, e.g. ``"$1,234.56"`` or ``"-$29.24"``.
        Returns ``"—"`` for NaN values.

    Examples
    --------
    >>> fmt_dollar(1234.56)
    '$1,234.56'
    >>> fmt_dollar(-29.24)
    '-$29.24'
    >>> fmt_dollar(float("nan"))
    '—'
    """
    if np.isnan(value):
        return "—"
    if sign and value < 0:
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """
    Format a float as a percentage string.

    Parameters
    ----------
    value : float
        Percentage value, e.g. ``52.38`` (not ``0.5238``).
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted string, e.g. ``"52.38%"``. Returns ``"—"`` for NaN.

    Examples
    --------
    >>> fmt_pct(52.38)
    '52.38%'
    >>> fmt_pct(float("nan"))
    '—'
    """
    return "—" if np.isnan(value) else f"{value:.{decimals}f}%"


def fmt_ratio(value: float, decimals: int = 3) -> str:
    """
    Format a ratio, returning ``"—"`` for NaN or Inf.

    Parameters
    ----------
    value : float
        Ratio value.
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        Formatted string, e.g. ``"3.099"``. Returns ``"—"`` for NaN/Inf.

    Examples
    --------
    >>> fmt_ratio(3.099)
    '3.099'
    >>> fmt_ratio(float("inf"))
    '—'
    >>> fmt_ratio(float("nan"))
    '—'
    """
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "—"
    return f"{value:.{decimals}f}"


# ── Validation ────────────────────────────────────────────────────────────────

REQUIRED_INTERNAL_COLS: frozenset[str] = frozenset({
    "trade_id", "symbol", "exit_time", "gross_pnl",
    "fees", "commissions", "direction",
})


def validate_dataframe(
    df: pd.DataFrame,
    required: frozenset[str] | None = None,
) -> None:
    """
    Assert a cleaned trade DataFrame has the minimum required columns and rows.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned trade DataFrame.
    required : frozenset[str], optional
        Column names that must be present. Defaults to
        :data:`REQUIRED_INTERNAL_COLS`.

    Raises
    ------
    TradeDataError
        If any required column is absent, or if the DataFrame is empty.
    """
    required = required or REQUIRED_INTERNAL_COLS
    missing  = required - frozenset(df.columns)
    if missing:
        raise TradeDataError(
            f"Trade DataFrame is missing required columns: {sorted(missing)}. "
            f"Check that your CSV uses the expected Topstep export format."
        )
    if df.empty:
        raise TradeDataError("Trade DataFrame is empty — no trades found in CSV.")
