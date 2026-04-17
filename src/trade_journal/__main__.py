"""
__main__.py
CLI entry point for the trade-journal package.

Usage
-----
    python -m trade_journal --csv data/trades.csv --account 50000
    python -m trade_journal --csv data/trades.csv --account 50000 --output output/dashboard.html
    python -m trade_journal --csv data/trades.csv --bullets-only
    python -m trade_journal --version
    python -m trade_journal --csv data/trades.csv --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .analytics import TradeAnalytics
from .config import ACCOUNT_SIZE, OUTPUT_DIR
from .report import build
from .utils import TradeDataError


_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _configure_logging(level_name: str) -> None:
    """
    Configure root logger with a structured format.

    Parameters
    ----------
    level_name : str
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
    """
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level_name),
        stream=sys.stdout,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trade_journal",
        description="Bloomberg-style trading performance dashboard from Topstep CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m trade_journal --csv data/trades.csv --account 50000\n"
            "  python -m trade_journal --csv data/trades.csv --bullets-only\n"
            "  python -m trade_journal --version"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"trade-journal {__version__}",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        metavar="PATH",
        help="Path to Topstep trade export CSV.",
    )
    parser.add_argument(
        "--account",
        type=float,
        default=ACCOUNT_SIZE,
        metavar="SIZE",
        help=f"Account size in dollars (default: {ACCOUNT_SIZE:,.0f}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "dashboard.html",
        metavar="PATH",
        help="Output path for the HTML dashboard (default: output/dashboard.html).",
    )
    parser.add_argument(
        "--bullets-only",
        action="store_true",
        help="Print resume bullets to stdout only; skip dashboard generation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=_LOG_LEVELS,
        metavar="LEVEL",
        help=f"Logging verbosity: {', '.join(_LOG_LEVELS)} (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Main CLI entry point.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    log  = logging.getLogger(__name__)

    # ── Require --csv for non-version commands ────────────────────────────
    if args.csv is None:
        log.error("--csv is required.  Run with --help for usage.")
        return 1

    # ── Validate inputs before touching any data ──────────────────────────
    if not args.csv.exists():
        log.error("CSV file not found: %s", args.csv)
        return 1

    if args.account <= 0:
        log.error("Account size must be positive; got %s", args.account)
        return 1

    # ── Load data ─────────────────────────────────────────────────────────
    try:
        ta = TradeAnalytics(args.csv, account_size=args.account)
    except (TradeDataError, FileNotFoundError) as exc:
        log.error("Failed to load trade data: %s", exc)
        return 1

    # ── Bullets only ──────────────────────────────────────────────────────
    if args.bullets_only:
        ta.resume_bullets()
        return 0

    # ── Build dashboard ───────────────────────────────────────────────────
    try:
        out = build(ta, args.output)
        print(f"\n  Dashboard → {out}")
        print(f"  Open in browser: file://{out}")
        print(f"  Note: internet connection required (Plotly loads from CDN)\n")
        ta.resume_bullets()
    except TradeDataError as exc:
        log.error("Dashboard data error: %s", exc)
        return 1
    except OSError as exc:
        log.error("Could not write dashboard file: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
