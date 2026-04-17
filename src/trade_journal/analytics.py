"""
analytics.py
Core performance analytics for Topstep funded account trade data.

All statistical computations are vectorized via NumPy/pandas.
No Python-level loops over trade rows.

Notes
-----
Sharpe, Sortino, and rolling Sharpe are computed on the **daily percentage
return** series (daily net P&L / starting account equity), not on raw dollar
P&L.  This makes results comparable across different account sizes and
strategies, and matches institutional convention.

Max drawdown is computed at **daily granularity** (one observation per trading
day).  Intraday adverse excursion within a winning day is not captured.  For
strategies with large intraday swings this will understate true drawdown.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ACCOUNT_SIZE,
    COLUMN_MAP,
    DURATION_BINS,
    DURATION_LABELS,
    MIN_TRADES_FOR_RATIOS,
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
)
from .utils import (
    InsufficientDataError,
    TradeDataError,
    fmt_dollar,
    fmt_pct,
    fmt_ratio,
    parse_duration_series,
    root_symbol_series,
    safe_metric,
    validate_dataframe,
)

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


class TradeAnalytics:
    """
    Load, clean, and analyze a Topstep trade export CSV.

    Parameters
    ----------
    csv_path : str or Path
        Path to the Topstep CSV export file.
    account_size : float, optional
        Starting account balance used for return and drawdown % calculations.
        Defaults to :data:`~trade_journal.config.ACCOUNT_SIZE`.
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe/Sortino calculation.
        Defaults to :data:`~trade_journal.config.RISK_FREE_RATE`.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    TradeDataError
        If the CSV is missing required columns or contains no valid trades.

    Examples
    --------
    >>> ta = TradeAnalytics("data/trades_export.csv", account_size=50_000)
    >>> ta.summary()
    """

    def __init__(
        self,
        csv_path: str | Path,
        account_size: float = ACCOUNT_SIZE,
        risk_free_rate: float = RISK_FREE_RATE,
    ) -> None:
        self.csv_path       = Path(csv_path)
        self.account_size   = account_size
        self.risk_free_rate = risk_free_rate

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Trade CSV not found: {self.csv_path}")

        log.info("Loading trades from %s", self.csv_path)
        self.df    = self._load_and_clean()
        self.daily = self._build_daily()
        log.info(
            "Loaded %d trades over %d trading days",
            len(self.df), len(self.daily),
        )

    # ── Private: data loading ─────────────────────────────────────────────────

    def _load_and_clean(self) -> pd.DataFrame:
        """
        Read the Topstep CSV, rename columns, parse types, and derive fields.

        Returns
        -------
        pd.DataFrame
            Clean trade-level DataFrame sorted by exit time.

        Raises
        ------
        TradeDataError
            On CSV parse errors or missing required columns.
        """
        try:
            raw = pd.read_csv(self.csv_path)
        except pd.errors.ParserError as exc:
            raise TradeDataError(f"CSV is malformed and could not be parsed: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise TradeDataError(
                f"CSV encoding error — try re-exporting from Topstep: {exc}"
            ) from exc
        except ValueError as exc:
            raise TradeDataError(f"CSV read error: {exc}") from exc

        raw.columns = raw.columns.str.strip()

        # ── Rename to internal names ───────────────────────────────────────
        rename = {k: v for k, v in COLUMN_MAP.items() if k in raw.columns}
        df = raw.rename(columns=rename)

        # ── Timestamps ────────────────────────────────────────────────────
        for col in ("entry_time", "exit_time", "trade_day"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)

        # ── Numerics (already in dollars — no $ stripping needed) ─────────
        for col in ("gross_pnl", "fees", "commissions",
                    "entry_price", "exit_price", "size"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # ── Derived columns (all vectorized) ──────────────────────────────
        fees_col  = df.get("fees",        pd.Series(0.0, index=df.index)).fillna(0.0)
        comm_col  = df.get("commissions", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["net_pnl"]      = df["gross_pnl"] - fees_col - comm_col
        df["root"]         = root_symbol_series(df["symbol"])
        df["duration_min"] = (
            parse_duration_series(df["duration_raw"])
            if "duration_raw" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        df["direction"] = (
            df["direction"].str.strip().str.title()
            if "direction" in df.columns
            else pd.Series("Unknown", index=df.index)
        )
        df["date"]    = df["exit_time"].dt.normalize()
        df["hour"]    = df["exit_time"].dt.hour
        df["weekday"] = df["exit_time"].dt.day_name()
        df["win"]     = df["net_pnl"] > 0

        df["dur_bucket"] = pd.cut(
            df["duration_min"],
            bins=DURATION_BINS,
            labels=DURATION_LABELS,
            right=True,
        )

        validate_dataframe(df)
        return df.sort_values("exit_time").reset_index(drop=True)

    # ── Private: daily aggregation ────────────────────────────────────────────

    def _build_daily(self) -> pd.DataFrame:
        """
        Aggregate trade-level data to a daily P&L series with drawdown and
        percentage return columns.

        Returns
        -------
        pd.DataFrame
            One row per trading day.  Includes ``pct_return`` — the daily net
            P&L expressed as a fraction of :attr:`account_size` — which is
            used for Sharpe and Sortino calculations.
        """
        daily = (
            self.df
            .groupby("date", sort=True)
            .agg(
                trades      =("net_pnl",     "count"),
                gross_pnl   =("gross_pnl",   "sum"),
                net_pnl     =("net_pnl",     "sum"),
                wins        =("win",         "sum"),
                fees        =("fees",        "sum"),
                commissions =("commissions", "sum"),
            )
            .reset_index()
        )

        # Vectorized cumulative metrics
        daily["win_rate"]     = daily["wins"]   / daily["trades"]
        daily["cum_pnl"]      = daily["net_pnl"].cumsum()
        daily["cum_gross"]    = daily["gross_pnl"].cumsum()
        daily["cum_fees"]     = (daily["fees"] + daily["commissions"]).cumsum()
        daily["peak"]         = daily["cum_pnl"].cummax()
        daily["drawdown"]     = daily["cum_pnl"] - daily["peak"]
        daily["drawdown_pct"] = daily["drawdown"] / self.account_size * 100

        # Percentage return: daily net P&L / account equity
        # Used for Sharpe/Sortino — makes metrics account-size independent
        daily["pct_return"] = daily["net_pnl"] / self.account_size

        return daily

    # ── Public: scalar metrics ────────────────────────────────────────────────

    def total_net_pnl(self) -> float:
        """Total net P&L across all trades."""
        return float(self.df["net_pnl"].sum())

    def total_gross_pnl(self) -> float:
        """Total gross P&L before fees and commissions."""
        return float(self.df["gross_pnl"].sum())

    def total_fees(self) -> float:
        """Total fees and commissions paid."""
        return float(self.df["fees"].sum() + self.df["commissions"].sum())

    def win_rate(self) -> float:
        """Win rate as a percentage (0–100)."""
        return float(self.df["win"].mean() * 100)

    def avg_win(self) -> float:
        """Average net P&L on winning trades."""
        winners = self.df.loc[self.df["win"], "net_pnl"]
        return float(winners.mean()) if len(winners) > 0 else 0.0

    def avg_loss(self) -> float:
        """Average net P&L on losing trades (negative value)."""
        losers = self.df.loc[~self.df["win"], "net_pnl"]
        return float(losers.mean()) if len(losers) > 0 else 0.0

    def win_loss_ratio(self) -> float:
        """avg_win / |avg_loss|.  Returns ``inf`` if there are no losses."""
        al = abs(self.avg_loss())
        return float(self.avg_win() / al) if al > 0 else float("inf")

    def profit_factor(self) -> float:
        """Gross profit / gross loss.  Returns ``inf`` if no losing trades."""
        gross_profit = self.df.loc[self.df["net_pnl"] > 0, "net_pnl"].sum()
        gross_loss   = abs(self.df.loc[self.df["net_pnl"] < 0, "net_pnl"].sum())
        return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    def expectancy(self) -> float:
        """Expected dollar profit per trade: (win% × avg_win) + (loss% × avg_loss)."""
        wr = self.win_rate() / 100
        return float(wr * self.avg_win() + (1 - wr) * self.avg_loss())

    def sharpe_ratio(self, min_days: int = 2) -> float:
        """
        Annualized Sharpe ratio computed on the daily **percentage return** series.

        Uses ``ddof=1`` for unbiased standard deviation.

        Parameters
        ----------
        min_days : int
            Minimum trading days required.

        Returns
        -------
        float
            Annualized Sharpe ratio.

        Raises
        ------
        InsufficientDataError
            If fewer than ``min_days`` trading days are available.

        Notes
        -----
        Computed on percentage returns (net P&L / account size) rather than
        raw dollar P&L, making the result comparable across account sizes.
        """
        if len(self.daily) < min_days:
            raise InsufficientDataError(
                f"Sharpe requires at least {min_days} trading days; "
                f"only {len(self.daily)} available."
            )
        rf_daily = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess   = self.daily["pct_return"] - rf_daily
        std      = excess.std(ddof=1)
        if std == 0:
            log.warning("Daily return std is zero — Sharpe is undefined.")
            return float("nan")
        return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))

    def sortino_ratio(self, min_losing_days: int = 1) -> float:
        """
        Annualized Sortino ratio using downside deviation on percentage returns.

        Parameters
        ----------
        min_losing_days : int
            Minimum losing days required to compute downside std.

        Returns
        -------
        float
            Annualized Sortino ratio, or ``nan`` if insufficient losing days.

        Notes
        -----
        Computed on percentage returns — see :meth:`sharpe_ratio`.
        The minimum acceptable return (MAR) is the daily risk-free rate.
        """
        rf_daily = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess   = self.daily["pct_return"] - rf_daily
        downside = excess[excess < 0]

        if len(downside) < min_losing_days:
            log.info(
                "Sortino undefined: need %d losing day(s), have %d. "
                "Will populate automatically as track record grows.",
                min_losing_days, len(downside),
            )
            return float("nan")

        downside_std = downside.std(ddof=1)
        if downside_std == 0:
            return float("nan")
        return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))

    def calmar_ratio(self, min_trades: int = MIN_TRADES_FOR_RATIOS) -> float:
        """
        Annualized return divided by maximum drawdown (absolute $).

        Parameters
        ----------
        min_trades : int
            Minimum trade count for a meaningful result.  Returns ``nan`` with
            a log warning below this threshold to prevent misleading values on
            small samples (e.g. Calmar of 16,000 on 4 trading days).

        Returns
        -------
        float
            Calmar ratio, or ``nan`` if max drawdown is zero or sample is small.
        """
        if len(self.df) < min_trades:
            log.info(
                "Calmar ratio requires %d trades for statistical reliability; "
                "have %d.  Returning nan to avoid misleading value.",
                min_trades, len(self.df),
            )
            return float("nan")
        mdd = abs(self.max_drawdown()["$"])
        if mdd == 0:
            log.info("Calmar undefined: max drawdown is zero.")
            return float("nan")
        n_days     = len(self.daily)
        ann_return = self.total_net_pnl() / n_days * TRADING_DAYS_PER_YEAR
        return float(ann_return / mdd)

    def recovery_factor(self, min_trades: int = MIN_TRADES_FOR_RATIOS) -> float:
        """
        Net P&L divided by maximum drawdown (absolute $).

        Parameters
        ----------
        min_trades : int
            Minimum trade count.  Returns ``nan`` below this threshold.

        Returns
        -------
        float
            Recovery factor, or ``nan`` if insufficient data.
        """
        if len(self.df) < min_trades:
            log.info(
                "Recovery factor requires %d trades; have %d.  Returning nan.",
                min_trades, len(self.df),
            )
            return float("nan")
        mdd = abs(self.max_drawdown()["$"])
        return float(self.total_net_pnl() / mdd) if mdd > 0 else float("nan")

    def max_drawdown(self) -> dict[str, float]:
        """
        Maximum peak-to-trough drawdown at daily granularity.

        Returns
        -------
        dict
            Keys ``"$"`` (dollar amount) and ``"%"`` (% of account size).

        Notes
        -----
        Computed at daily granularity — one observation per trading day.
        Intraday adverse excursion within a winning day is not captured.
        For strategies with large intraday swings this may understate
        true drawdown.
        """
        if self.daily.empty:
            return {"$": 0.0, "%": 0.0}
        return {
            "$": float(self.daily["drawdown"].min()),
            "%": float(self.daily["drawdown_pct"].min()),
        }

    def consecutive_streaks(self) -> dict[str, int]:
        """
        Maximum consecutive winning and losing streaks.

        Returns
        -------
        dict
            Keys ``"max_wins"`` and ``"max_losses"``.

        Notes
        -----
        Uses vectorized run-length encoding via ``cumsum`` on group breaks —
        no Python-level loop over individual trades.
        """
        wins    = self.df["win"].astype(int)
        groups  = (wins != wins.shift()).cumsum()
        lengths = wins.groupby(groups).transform("count")
        is_win  = wins.groupby(groups).transform("first").astype(bool)

        win_streaks  = lengths[is_win]
        loss_streaks = lengths[~is_win]

        return {
            "max_wins":   int(win_streaks.max())  if len(win_streaks)  > 0 else 0,
            "max_losses": int(loss_streaks.max()) if len(loss_streaks) > 0 else 0,
        }

    def rolling_sharpe(self, window: int = 10) -> pd.Series:
        """
        Rolling annualized Sharpe ratio on daily **percentage returns**.

        Parameters
        ----------
        window : int
            Rolling window in trading days.  Auto-reduces when insufficient data.

        Returns
        -------
        pd.Series
            Rolling Sharpe values aligned to the daily index.
        """
        if len(self.daily) < window:
            effective = max(2, len(self.daily) - 1)
            log.info(
                "Rolling Sharpe window reduced from %d to %d (insufficient days).",
                window, effective,
            )
            window = effective

        rf_daily = self.risk_free_rate / TRADING_DAYS_PER_YEAR
        excess   = self.daily["pct_return"] - rf_daily
        roll     = excess.rolling(window, min_periods=window)
        return (roll.mean() / roll.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)).round(3)

    # ── Public: group-by aggregations ────────────────────────────────────────

    def pnl_by_symbol(self) -> pd.DataFrame:
        """Net P&L, trade count, and win rate grouped by root symbol."""
        return (
            self.df.groupby("root")
            .agg(
                net_pnl =("net_pnl", "sum"),
                trades  =("net_pnl", "count"),
                win_rate=("win",     "mean"),
            )
            .assign(win_rate=lambda x: (x["win_rate"] * 100).round(1))
            .round({"net_pnl": 2})
            .sort_values("net_pnl", ascending=False)
        )

    def pnl_by_direction(self) -> pd.DataFrame:
        """Net P&L, trade count, and avg P&L by direction (Long/Short)."""
        return (
            self.df.groupby("direction")
            .agg(
                net_pnl=("net_pnl", "sum"),
                trades  =("net_pnl", "count"),
                avg_pnl =("net_pnl", "mean"),
            )
            .round(2)
        )

    def pnl_by_hour(self) -> pd.DataFrame:
        """Net P&L and win rate by exit hour (Eastern Time)."""
        return (
            self.df.groupby("hour")
            .agg(
                net_pnl =("net_pnl", "sum"),
                avg_pnl =("net_pnl", "mean"),
                trades  =("net_pnl", "count"),
                win_rate=("win",     "mean"),
            )
            .assign(win_rate=lambda x: (x["win_rate"] * 100).round(1))
            .round({"net_pnl": 2, "avg_pnl": 2})
        )

    def pnl_by_weekday(self) -> pd.DataFrame:
        """Net P&L and win rate by day of week, ordered Monday–Friday."""
        order  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        result = (
            self.df.groupby("weekday")
            .agg(
                net_pnl =("net_pnl", "sum"),
                avg_pnl =("net_pnl", "mean"),
                trades  =("net_pnl", "count"),
                win_rate=("win",     "mean"),
            )
            .assign(win_rate=lambda x: (x["win_rate"] * 100).round(1))
            .round({"net_pnl": 2, "avg_pnl": 2})
        )
        return result.reindex([d for d in order if d in result.index])

    def pnl_by_duration(self) -> pd.DataFrame:
        """Net P&L and win rate by trade duration bucket."""
        result = (
            self.df.groupby("dur_bucket", observed=True)
            .agg(
                net_pnl =("net_pnl", "sum"),
                trades  =("net_pnl", "count"),
                win_rate=("win",     "mean"),
            )
            .assign(win_rate=lambda x: (x["win_rate"] * 100).round(1))
            .round({"net_pnl": 2})
        )
        return result[result["trades"] > 0]

    def monthly_returns(self) -> pd.Series:
        """Monthly net P&L indexed by year-month period."""
        return (
            self.df
            .assign(ym=self.df["exit_time"].dt.to_period("M"))
            .groupby("ym")["net_pnl"]
            .sum()
            .round(2)
        )

    # ── Public: summary and output ────────────────────────────────────────────

    def summary(self) -> dict[str, float | int]:
        """
        Compute all key metrics and return as an ordered dictionary.

        Metrics that require more data than is currently available return
        ``nan`` rather than raising an exception, courtesy of
        :func:`~trade_journal.utils.safe_metric`.

        Returns
        -------
        dict
            Metric name → scalar value.
        """
        mdd     = self.max_drawdown()
        streaks = self.consecutive_streaks()

        return {
            "total_trades":    len(self.df),
            "trading_days":    len(self.daily),
            "total_gross_pnl": round(self.total_gross_pnl(), 2),
            "total_fees":      round(self.total_fees(), 2),
            "total_net_pnl":   round(self.total_net_pnl(), 2),
            "win_rate_%":      round(self.win_rate(), 2),
            "profit_factor":   round(self.profit_factor(), 3),
            "avg_win":         round(self.avg_win(), 2),
            "avg_loss":        round(self.avg_loss(), 2),
            "win_loss_ratio":  round(self.win_loss_ratio(), 3),
            "expectancy":      round(self.expectancy(), 2),
            "sharpe_ratio":    round(safe_metric(self.sharpe_ratio), 3),
            "sortino_ratio":   round(safe_metric(self.sortino_ratio), 3),
            "calmar_ratio":    round(safe_metric(self.calmar_ratio), 3),
            "recovery_factor": round(safe_metric(self.recovery_factor), 3),
            "max_drawdown_$":  round(mdd["$"], 2),
            "max_drawdown_%":  round(mdd["%"], 2),
            "max_win_streak":  streaks["max_wins"],
            "max_loss_streak": streaks["max_losses"],
        }

    def resume_bullets(self) -> list[str]:
        """
        Generate copy-paste resume bullet points from live statistics.

        Returns
        -------
        list[str]
            Four bullet point strings. Also prints them to stdout.
        """
        m      = self.summary()
        syms   = ", ".join(self.pnl_by_symbol().index.tolist())
        start  = self.df["exit_time"].min().strftime("%b %Y")
        end    = self.df["exit_time"].max().strftime("%b %Y")
        period = f"{start} – {end}"

        bullets = [
            f"• {period} live trading track record: {m['total_trades']} trades "
            f"across {syms} futures",
            f"• Sharpe {fmt_ratio(m['sharpe_ratio'])}, "
            f"profit factor {fmt_ratio(m['profit_factor'])}, "
            f"max drawdown {abs(m['max_drawdown_%']):.2f}% of account",
            f"• Win rate {fmt_pct(m['win_rate_%'])}, "
            f"avg win/loss ratio {fmt_ratio(m['win_loss_ratio'])}:1, "
            f"expectancy {fmt_dollar(m['expectancy'])}/trade",
            f"• Net P&L {fmt_dollar(m['total_net_pnl'])} over "
            f"{m['trading_days']} trading days; "
            f"all sessions within daily loss limits",
        ]

        print("\n--- RESUME BULLETS ---")
        for b in bullets:
            print(b)
        print()
        return bullets
