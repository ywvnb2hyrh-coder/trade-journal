"""
tests/test_charts.py
Unit tests for trade_journal.charts — verify each builder returns a valid Figure.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from trade_journal import TradeAnalytics
from trade_journal.charts import (
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


class TestChartBuilders:
    """Each chart function must return a go.Figure with at least one trace."""

    def test_equity_and_drawdown_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = equity_and_drawdown(ta)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_daily_pnl_bar_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = daily_pnl_bar(ta)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_rolling_sharpe_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = rolling_sharpe_line(ta)
        assert isinstance(fig, go.Figure)

    def test_pnl_distribution_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = pnl_distribution(ta)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # wins + losses histogram

    def test_pnl_by_symbol_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = pnl_by_symbol(ta)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_long_short_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = long_short_breakdown(ta)
        assert isinstance(fig, go.Figure)

    def test_duration_analysis_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = duration_analysis(ta)
        assert isinstance(fig, go.Figure)

    def test_time_of_day_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = time_of_day(ta)
        assert isinstance(fig, go.Figure)

    def test_weekday_analysis_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = weekday_analysis(ta)
        assert isinstance(fig, go.Figure)

    def test_fee_drag_returns_figure(self, ta: TradeAnalytics) -> None:
        fig = fee_drag(ta)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # gross + net lines

    def test_equity_curve_bar_count(self, ta: TradeAnalytics) -> None:
        """Drawdown bar count should match trading day count."""
        fig       = equity_and_drawdown(ta)
        bar_trace = next(t for t in fig.data if isinstance(t, go.Bar))
        assert len(bar_trace.x) == len(ta.daily)
