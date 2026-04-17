"""
tests/test_report.py
Verify the HTML report contains expected panel titles and metric values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trade_journal.analytics import TradeAnalytics
from trade_journal.report import build


@pytest.fixture
def dashboard_html(ta: TradeAnalytics, tmp_path: Path) -> str:
    """Build the dashboard and return the HTML string."""
    out  = tmp_path / "dashboard.html"
    path = build(ta, out)
    return path.read_text(encoding="utf-8")


class TestReportOutput:
    def test_html_file_created(self, ta: TradeAnalytics, tmp_path: Path) -> None:
        out = tmp_path / "test_dashboard.html"
        build(ta, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_contains_panel_titles(self, dashboard_html: str) -> None:
        for title in [
            "Equity curve",
            "Daily net P&L",
            "Rolling Sharpe",
            "P&L distribution",
            "Long vs short",
            "Trade duration",
            "Time of day",
            "Day of week",
            "fee drag",
        ]:
            assert title in dashboard_html, f"Panel title missing: {title}"

    def test_contains_metric_values(self, dashboard_html: str) -> None:
        # Win rate 60.0% should appear somewhere in the cards
        assert "60.00%" in dashboard_html

    def test_contains_plotly_script(self, dashboard_html: str) -> None:
        assert "plotly" in dashboard_html.lower()

    def test_contains_topstep_attribution(self, dashboard_html: str) -> None:
        assert "TOPSTEP" in dashboard_html
