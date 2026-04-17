"""
conftest.py
Shared pytest fixtures available to all test modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Canonical sample CSV ──────────────────────────────────────────────────────
# Three trading days, mix of wins/losses across two symbols.
# Hand-calculated expected values are derived from these exact rows.

SAMPLE_CSV = """\
Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions
1001,MCLK6,04/13/2026 09:00:00,04/13/2026 09:15:00,100.0,101.0,1.0,100.0,1,Long,04/13/2026,00:15:00,0.5
1002,MCLK6,04/13/2026 10:00:00,04/13/2026 10:10:00,101.0,100.5,1.0,-50.0,1,Short,04/13/2026,00:10:00,0.5
1003,MNQM6,04/14/2026 09:00:00,04/14/2026 09:30:00,20000.0,20050.0,0.5,100.0,1,Long,04/14/2026,00:30:00,0.5
1004,MNQM6,04/14/2026 10:00:00,04/14/2026 10:05:00,20050.0,20000.0,0.5,-50.0,1,Short,04/14/2026,00:05:00,0.5
1005,MCLK6,04/15/2026 09:00:00,04/15/2026 09:45:00,99.0,101.0,1.0,200.0,1,Long,04/15/2026,00:45:00,0.5
"""

# Hand-calculated expected values for the sample above
# net_pnl per trade: 98.5, -51.5, 99.0, -51.0, 198.5
# daily net_pnl: day1=47.0, day2=48.0, day3=198.5
# total_net_pnl: 293.5
# total_gross_pnl: 300.0
# total_fees: 6.5
# wins: trades 1001, 1003, 1005  (3 out of 5)
# win_rate: 60.0%
EXPECTED = {
    "total_net_pnl":   293.5,
    "total_gross_pnl": 300.0,
    "total_fees":        6.5,
    "win_rate_%":       60.0,
    "trade_count":        5,
    "trading_days":       3,
    # Day net_pnl: [47.0, 48.0, 198.5]
    # pct_return on $50k: [0.00094, 0.00096, 0.00397]
    # mean excess ≈ mean pct_return - rf_daily
    # annualised sharpe computed in test_sharpe_value
}


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Write the canonical sample CSV to a temp file and return the path."""
    p = tmp_path / "trades.csv"
    p.write_text(SAMPLE_CSV)
    return p


@pytest.fixture
def ta(csv_path: Path):
    """Return a loaded TradeAnalytics instance from the canonical sample."""
    from trade_journal.analytics import TradeAnalytics
    return TradeAnalytics(csv_path, account_size=50_000)
