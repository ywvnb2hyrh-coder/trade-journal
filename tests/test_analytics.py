"""
tests/test_analytics.py
Unit tests for trade_journal.analytics.TradeAnalytics.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_journal.analytics import TradeAnalytics
from trade_journal.utils import InsufficientDataError, TradeDataError

# ── Additional CSV fixtures ───────────────────────────────────────────────────

EXPECTED: dict[str, float | int] = {
    "total_net_pnl":   293.5,
    "total_gross_pnl": 300.0,
    "total_fees":        6.5,
    "win_rate_%":       60.0,
    "trade_count":        5,
    "trading_days":       3,
}



SINGLE_DAY_CSV = """\
Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions
9001,MCLK6,04/13/2026 09:00:00,04/13/2026 09:15:00,100.0,101.0,1.0,100.0,1,Long,04/13/2026,00:15:00,0.5
"""

ALL_WINS_CSV = """\
Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions
2001,MCLK6,04/13/2026 09:00:00,04/13/2026 09:30:00,100.0,101.0,0.5,100.0,1,Long,04/13/2026,00:30:00,0.5
2002,MCLK6,04/14/2026 09:00:00,04/14/2026 09:30:00,100.0,102.0,0.5,200.0,1,Long,04/14/2026,00:30:00,0.5
"""

EMPTY_CSV = (
    "Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,"
    "Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions\n"
)

NAN_PNL_CSV = """\
Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions
3001,MCLK6,04/13/2026 09:00:00,04/13/2026 09:15:00,100.0,101.0,1.0,,1,Long,04/13/2026,00:15:00,0.5
3002,MCLK6,04/13/2026 10:00:00,04/13/2026 10:10:00,101.0,100.5,1.0,50.0,1,Short,04/13/2026,00:10:00,0.5
"""

DUPE_ID_CSV = """\
Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions
5001,MCLK6,04/13/2026 09:00:00,04/13/2026 09:15:00,100.0,101.0,1.0,100.0,1,Long,04/13/2026,00:15:00,0.5
5001,MCLK6,04/13/2026 10:00:00,04/13/2026 10:10:00,101.0,100.5,1.0,-50.0,1,Short,04/13/2026,00:10:00,0.5
"""

EXTRA_COLS_CSV = """\
Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions,UnknownCol,AnotherExtra
4001,MCLK6,04/13/2026 09:00:00,04/13/2026 09:15:00,100.0,101.0,1.0,100.0,1,Long,04/13/2026,00:15:00,0.5,foo,bar
4002,MCLK6,04/14/2026 09:00:00,04/14/2026 09:15:00,100.0,101.0,1.0,50.0,1,Long,04/14/2026,00:15:00,0.5,baz,qux
"""

BAD_TIMESTAMP_CSV = """\
Id,ContractName,EnteredAt,ExitedAt,EntryPrice,ExitPrice,Fees,PnL,Size,Type,TradeDay,TradeDuration,Commissions
6001,MCLK6,NOT_A_DATE,04/13/2026 09:15:00,100.0,101.0,1.0,100.0,1,Long,04/13/2026,00:15:00,0.5
6002,MCLK6,04/13/2026 10:00:00,ALSO_BAD,101.0,100.5,1.0,-50.0,1,Short,04/13/2026,00:10:00,0.5
"""


def _write(tmp_path: Path, content: str, name: str = "t.csv") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ── Data loading ──────────────────────────────────────────────────────────────

class TestDataLoading:
    def test_row_count(self, ta: TradeAnalytics) -> None:
        assert len(ta.df) == EXPECTED["trade_count"]

    def test_required_columns_present(self, ta: TradeAnalytics) -> None:
        for col in ("trade_id", "symbol", "root", "net_pnl", "win",
                    "duration_min", "date", "hour", "weekday"):
            assert col in ta.df.columns, f"Missing column: {col}"

    def test_root_symbols_parsed(self, ta: TradeAnalytics) -> None:
        assert set(ta.df["root"].unique()) == {"MCL", "MNQ"}

    def test_net_pnl_subtracts_fees(self, ta: TradeAnalytics) -> None:
        # trade 1001: gross=100, fees=1.0, commissions=0.5 → net=98.5
        row = ta.df[ta.df["trade_id"] == 1001].iloc[0]
        assert row["net_pnl"] == pytest.approx(98.5)

    def test_win_flag(self, ta: TradeAnalytics) -> None:
        wins  = set(ta.df.loc[ta.df["win"],  "trade_id"])
        loses = set(ta.df.loc[~ta.df["win"], "trade_id"])
        assert 1001 in wins
        assert 1003 in wins
        assert 1005 in wins
        assert 1002 in loses
        assert 1004 in loses

    def test_pct_return_in_daily(self, ta: TradeAnalytics) -> None:
        # pct_return lives in the daily aggregation, not trade-level df
        assert "pct_return" in ta.daily.columns
        expected = ta.daily["net_pnl"] / ta.account_size
        pd.testing.assert_series_equal(
            ta.daily["pct_return"].round(8),
            expected.round(8),
            check_names=False,
        )

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TradeAnalytics(tmp_path / "ghost.csv")

    def test_empty_csv_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TradeDataError):
            TradeAnalytics(_write(tmp_path, EMPTY_CSV))

    def test_nan_pnl_treated_as_zero(self, tmp_path: Path) -> None:
        # NaN in PnL column should be coerced to 0.0, not crash
        ta = TradeAnalytics(_write(tmp_path, NAN_PNL_CSV), account_size=50_000)
        assert len(ta.df) == 2

    def test_duplicate_trade_ids_load(self, tmp_path: Path) -> None:
        # Duplicate IDs should load without error (Topstep can export them)
        ta = TradeAnalytics(_write(tmp_path, DUPE_ID_CSV), account_size=50_000)
        assert len(ta.df) == 2

    def test_extra_columns_ignored(self, tmp_path: Path) -> None:
        ta = TradeAnalytics(_write(tmp_path, EXTRA_COLS_CSV), account_size=50_000)
        assert len(ta.df) == 2

    def test_bad_timestamps_coerced(self, tmp_path: Path) -> None:
        # Malformed timestamps should coerce to NaT without crashing
        ta = TradeAnalytics(_write(tmp_path, BAD_TIMESTAMP_CSV), account_size=50_000)
        assert len(ta.df) == 2


# ── Scalar metrics ────────────────────────────────────────────────────────────

class TestMetrics:
    def test_total_net_pnl(self, ta: TradeAnalytics) -> None:
        assert ta.total_net_pnl() == pytest.approx(EXPECTED["total_net_pnl"])

    def test_total_gross_pnl(self, ta: TradeAnalytics) -> None:
        assert ta.total_gross_pnl() == pytest.approx(EXPECTED["total_gross_pnl"])

    def test_total_fees(self, ta: TradeAnalytics) -> None:
        assert ta.total_fees() == pytest.approx(EXPECTED["total_fees"])

    def test_win_rate(self, ta: TradeAnalytics) -> None:
        assert ta.win_rate() == pytest.approx(EXPECTED["win_rate_%"])

    def test_win_rate_range(self, ta: TradeAnalytics) -> None:
        assert 0.0 <= ta.win_rate() <= 100.0

    def test_profit_factor_positive(self, ta: TradeAnalytics) -> None:
        assert ta.profit_factor() > 0

    def test_profit_factor_formula(self, ta: TradeAnalytics) -> None:
        gross_profit = ta.df.loc[ta.df["net_pnl"] > 0, "net_pnl"].sum()
        gross_loss   = abs(ta.df.loc[ta.df["net_pnl"] < 0, "net_pnl"].sum())
        expected     = gross_profit / gross_loss
        assert ta.profit_factor() == pytest.approx(expected, rel=1e-4)

    def test_avg_win_positive(self, ta: TradeAnalytics) -> None:
        assert ta.avg_win() > 0

    def test_avg_loss_negative(self, ta: TradeAnalytics) -> None:
        assert ta.avg_loss() < 0

    def test_expectancy_positive(self, ta: TradeAnalytics) -> None:
        assert ta.expectancy() > 0

    def test_expectancy_formula(self, ta: TradeAnalytics) -> None:
        wr       = ta.win_rate() / 100
        expected = wr * ta.avg_win() + (1 - wr) * ta.avg_loss()
        assert ta.expectancy() == pytest.approx(expected, rel=1e-4)

    def test_sharpe_requires_min_days(self, tmp_path: Path) -> None:
        ta = TradeAnalytics(_write(tmp_path, SINGLE_DAY_CSV), account_size=50_000)
        with pytest.raises(InsufficientDataError):
            ta.sharpe_ratio()

    def test_sharpe_computed_on_pct_returns(self, ta: TradeAnalytics) -> None:
        """
        Verify Sharpe uses percentage returns, not dollar P&L.

        Hand-calculated from sample data:
          daily net_pnl:  [47.0, 48.0, 198.5]
          pct_return:     [0.00094, 0.00096, 0.00397]  (/ 50_000)
          rf_daily:       0.0525 / 252 ≈ 0.000208333
          excess:         pct_return - rf_daily
          sharpe:         mean(excess) / std(excess, ddof=1) * sqrt(252)
        """
        rf_daily  = 0.0525 / 252
        pct       = ta.daily["pct_return"].values
        excess    = pct - rf_daily
        expected  = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252))
        assert ta.sharpe_ratio() == pytest.approx(expected, rel=1e-4)

    def test_sortino_nan_without_losing_days(self, tmp_path: Path) -> None:
        ta = TradeAnalytics(_write(tmp_path, ALL_WINS_CSV), account_size=50_000)
        assert math.isnan(ta.sortino_ratio())

    def test_sortino_nan_with_all_positive_days(self, ta: TradeAnalytics) -> None:
        # All three trading days in sample are net positive → Sortino is nan
        # (no losing days means downside std cannot be computed)
        # Sortino will populate as the track record grows and losing days appear
        result = ta.sortino_ratio()
        assert math.isnan(result)

    def test_calmar_nan_below_min_trades(self, ta: TradeAnalytics) -> None:
        # Sample has 5 trades, min is 50 → should return nan
        assert math.isnan(ta.calmar_ratio())

    def test_recovery_nan_below_min_trades(self, ta: TradeAnalytics) -> None:
        assert math.isnan(ta.recovery_factor())

    def test_max_drawdown_non_positive(self, ta: TradeAnalytics) -> None:
        mdd = ta.max_drawdown()
        assert mdd["$"] <= 0
        assert mdd["%"] <= 0

    def test_consecutive_streaks_types(self, ta: TradeAnalytics) -> None:
        s = ta.consecutive_streaks()
        assert isinstance(s["max_wins"],   int)
        assert isinstance(s["max_losses"], int)
        assert s["max_wins"]   >= 1
        assert s["max_losses"] >= 1

    def test_consecutive_streaks_vectorized(self, ta: TradeAnalytics) -> None:
        # Verify result matches a naive loop implementation
        wins = ta.df["win"].tolist()
        best_w = best_l = cur_w = cur_l = 0
        for w in wins:
            if w:
                cur_w += 1; cur_l = 0
            else:
                cur_l += 1; cur_w = 0
            best_w = max(best_w, cur_w)
            best_l = max(best_l, cur_l)
        s = ta.consecutive_streaks()
        assert s["max_wins"]   == best_w
        assert s["max_losses"] == best_l

    def test_rolling_sharpe_length(self, ta: TradeAnalytics) -> None:
        rs = ta.rolling_sharpe(window=2)
        assert len(rs) == len(ta.daily)

    def test_rolling_sharpe_uses_pct_returns(self, ta: TradeAnalytics) -> None:
        # Rolling Sharpe should differ from dollar-based version
        rs_pct  = ta.rolling_sharpe(window=2)
        # Verify no raw dollar magnitudes appear (pct values should be < 100)
        valid   = rs_pct.dropna()
        assert (valid.abs() < 1000).all(), "Rolling Sharpe looks like dollar-based"


# ── Group aggregations ────────────────────────────────────────────────────────

class TestGroupAggregations:
    def test_pnl_by_symbol_symbols(self, ta: TradeAnalytics) -> None:
        assert set(ta.pnl_by_symbol().index) == {"MCL", "MNQ"}

    def test_pnl_by_symbol_sum(self, ta: TradeAnalytics) -> None:
        # Total across symbols should equal total net P&L
        total = ta.pnl_by_symbol()["net_pnl"].sum()
        assert total == pytest.approx(ta.total_net_pnl(), rel=1e-3)

    def test_pnl_by_direction_keys(self, ta: TradeAnalytics) -> None:
        assert set(ta.pnl_by_direction().index).issubset({"Long", "Short"})

    def test_pnl_by_hour_nonempty(self, ta: TradeAnalytics) -> None:
        assert not ta.pnl_by_hour().empty

    def test_pnl_by_duration_nonempty(self, ta: TradeAnalytics) -> None:
        assert not ta.pnl_by_duration().empty

    def test_monthly_returns_sum(self, ta: TradeAnalytics) -> None:
        assert ta.monthly_returns().sum() == pytest.approx(ta.total_net_pnl(), rel=1e-3)

    def test_pnl_by_weekday_ordered(self, ta: TradeAnalytics) -> None:
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        days  = ta.pnl_by_weekday().index.tolist()
        for d in days:
            assert d in order


# ── Summary dict ──────────────────────────────────────────────────────────────

class TestSummary:
    def test_returns_dict(self, ta: TradeAnalytics) -> None:
        assert isinstance(ta.summary(), dict)

    def test_required_keys_present(self, ta: TradeAnalytics) -> None:
        s = ta.summary()
        for key in ("total_net_pnl", "win_rate_%", "profit_factor",
                    "sharpe_ratio", "sortino_ratio", "max_drawdown_%"):
            assert key in s, f"Missing key: {key}"

    def test_low_sample_ratios_are_nan(self, ta: TradeAnalytics) -> None:
        s = ta.summary()
        # 5 trades < 50 minimum → calmar and recovery should be nan
        assert math.isnan(s["calmar_ratio"])
        assert math.isnan(s["recovery_factor"])
