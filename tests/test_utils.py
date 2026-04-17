"""
tests/test_utils.py
Unit tests for trade_journal.utils — parsers, formatters, validators.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trade_journal.utils import (
    InsufficientDataError,
    TradeDataError,
    fmt_dollar,
    fmt_pct,
    fmt_ratio,
    parse_duration_seconds,
    parse_duration_series,
    root_symbol,
    root_symbol_series,
    safe_metric,
    validate_dataframe,
)


# ── root_symbol ───────────────────────────────────────────────────────────────

class TestRootSymbol:
    @pytest.mark.parametrize("raw,expected", [
        ("MCLK6",   "MCL"),
        ("MNQM6",   "MNQ"),
        ("ESH25",   "ES"),
        ("NQU24",   "NQ"),
        ("CLZ3",    "CL"),
        ("GCM24",   "GC"),
        ("6EH25",   "6E"),
        ("/ESH25",  "ES"),
        ("/MCLK6",  "MCL"),
        ("MCL",     "MCL"),   # no suffix — passthrough
    ])
    def test_strips_suffix(self, raw: str, expected: str) -> None:
        assert root_symbol(raw) == expected

    @pytest.mark.parametrize("bad", ["", "   ", None, 123])
    def test_bad_input_returns_unknown(self, bad: object) -> None:
        assert root_symbol(bad) == "UNKNOWN"  # type: ignore[arg-type]

    def test_series_vectorized(self) -> None:
        s      = pd.Series(["MCLK6", "MNQM6", "/ESH25", None, ""])
        result = root_symbol_series(s)
        assert result.tolist() == ["MCL", "MNQ", "ES", "UNKNOWN", "UNKNOWN"]


# ── parse_duration_seconds ────────────────────────────────────────────────────

class TestParseDurationSeconds:
    @pytest.mark.parametrize("raw,expected", [
        ("00:15:00",           900.0),
        ("01:00:00",          3600.0),
        ("00:00:30",            30.0),
        ("00:11:09.8021060",   669.8021060),
        ("00:02:08.8827700",   128.8827700),
        ("00:34:16.2189400",  2056.2189400),
        ("12:00:00",         43200.0),
        ("00:00:01",             1.0),
    ])
    def test_standard_formats(self, raw: str, expected: float) -> None:
        assert parse_duration_seconds(raw) == pytest.approx(expected, rel=1e-5)

    @pytest.mark.parametrize("bad", [None, "", "   ", float("nan"), "notaduration"])
    def test_bad_input_returns_zero(self, bad: object) -> None:
        assert parse_duration_seconds(bad) == 0.0

    def test_series_returns_minutes(self) -> None:
        s      = pd.Series(["00:15:00", "01:00:00", ""])
        result = parse_duration_series(s)
        assert result.iloc[0] == pytest.approx(15.0, rel=1e-4)
        assert result.iloc[1] == pytest.approx(60.0, rel=1e-4)
        assert result.iloc[2] == pytest.approx(0.0)


# ── formatters ────────────────────────────────────────────────────────────────

class TestFmtDollar:
    @pytest.mark.parametrize("value,expected", [
        (1234.56,     "$1,234.56"),
        (0.0,         "$0.00"),
        (1_000_000.0, "$1,000,000.00"),
        (-29.24,      "-$29.24"),
        (-1234.56,    "-$1,234.56"),
    ])
    def test_valid_values(self, value: float, expected: str) -> None:
        assert fmt_dollar(value) == expected

    def test_nan_returns_dash(self) -> None:
        assert fmt_dollar(float("nan")) == "—"


class TestFmtPct:
    @pytest.mark.parametrize("value,decimals,expected", [
        (52.38, 2, "52.38%"),
        (100.0, 0, "100%"),
        (0.0,   2, "0.00%"),
        (3.141, 1, "3.1%"),
    ])
    def test_valid_values(self, value: float, decimals: int, expected: str) -> None:
        assert fmt_pct(value, decimals) == expected

    def test_nan_returns_dash(self) -> None:
        assert fmt_pct(float("nan")) == "—"


class TestFmtRatio:
    @pytest.mark.parametrize("value,expected_contains", [
        (3.099, "3.099"),
        (1.0,   "1.000"),
    ])
    def test_valid_values(self, value: float, expected_contains: str) -> None:
        assert fmt_ratio(value) == expected_contains

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_special_floats_return_dash(self, bad: float) -> None:
        assert fmt_ratio(bad) == "—"


# ── safe_metric ───────────────────────────────────────────────────────────────

class TestSafeMetric:
    def test_returns_value_on_success(self) -> None:
        assert safe_metric(lambda: 42.0) == pytest.approx(42.0)

    def test_returns_nan_on_insufficient_data(self) -> None:
        def raises() -> float:
            raise InsufficientDataError("not enough days")
        result = safe_metric(raises)
        assert math.isnan(result)

    def test_passes_args_through(self) -> None:
        assert safe_metric(lambda x, y: x + y, 3.0, 4.0) == pytest.approx(7.0)


# ── validate_dataframe ────────────────────────────────────────────────────────

class TestValidateDataframe:
    def test_valid_df_passes(self) -> None:
        df = pd.DataFrame({
            "trade_id": [1], "symbol": ["ES"], "exit_time": [pd.Timestamp.now()],
            "gross_pnl": [100.0], "fees": [1.0], "commissions": [0.5],
            "direction": ["Long"],
        })
        validate_dataframe(df)  # should not raise

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"trade_id": [1], "symbol": ["ES"]})
        with pytest.raises(TradeDataError, match="missing required columns"):
            validate_dataframe(df)

    def test_empty_df_raises(self) -> None:
        cols = ["trade_id", "symbol", "exit_time", "gross_pnl",
                "fees", "commissions", "direction"]
        df   = pd.DataFrame(columns=cols)
        with pytest.raises(TradeDataError, match="empty"):
            validate_dataframe(df)
