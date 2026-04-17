# trade-journal

> Quantitative performance analytics system built on live Topstep funded account data.

Automatically ingests Topstep's trade export CSV, computes institutional-grade performance metrics, adds a manual research journal layer, and generates a Bloomberg-terminal style interactive dashboard — updated after every trading session.

---

## Live track record snapshot

| Metric | Value |
|---|---|
| Instruments | MCL, MNQ futures |
| Track record | Apr 2026 – present |
| Total trades | 21 |
| Win rate | 52.38% |
| Profit factor | 3.40 |
| Avg win / avg loss | $90.34 / -$29.24 |
| Win/loss ratio | 3.09:1 |
| Expectancy | $33.40 / trade |
| Max drawdown | 0.01% of account |

*All figures net of commissions and fees. Track record verified through Topstep funded account platform.*

---

## Dashboard preview

![Dashboard]((image.png))

---

## Project structure

```
trade-journal/
├── src/trade_journal/
│   ├── config.py        # All constants, column mappings, color palette
│   ├── utils.py         # Pure parsing/formatting utilities + custom exceptions
│   ├── analytics.py     # Core metrics engine (fully vectorized)
│   ├── charts.py        # Interactive Plotly chart builders
│   ├── report.py        # HTML dashboard assembler
│   └── __main__.py      # CLI entry point with input validation
├── tests/
│   └── test_analytics.py  # 38 unit tests (pytest)
├── data/
│   └── sample_trades.csv  # Synthetic example — real file excluded via .gitignore
├── docs/
│   └── dashboard_preview.png
├── pyproject.toml
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/yourusername/trade-journal
cd trade-journal
pip3 install pandas numpy plotly pytest

# Export trades from Topstep → place CSV in data/
PYTHONPATH=src python3 -m trade_journal --csv data/your_trades.csv --account 50000

# Dashboard renders at output/dashboard.html — open in any browser
```

---

## What each module does

### `config.py`
Single source of truth for every constant in the project. Account size, risk-free rate, Topstep CSV column mappings, instrument tick sizes, duration bucket boundaries, and the Bloomberg color palette. Changing any value here propagates everywhere — nothing is hardcoded across files.

### `utils.py`
Pure, stateless utility functions with no side effects. Includes `root_symbol()` which strips futures contract month/year codes (MCLK6 → MCL), `parse_duration_seconds()` which handles Topstep's HH:MM:SS.ffffff format, dollar/percent/ratio formatters, and `validate_dataframe()` which checks for required columns before any computation runs. Also defines `TradeDataError` and `InsufficientDataError` — typed custom exceptions that give informative error messages instead of raw Python tracebacks.

### `analytics.py`
The core engine. Loads and cleans the Topstep CSV, computes all performance metrics using vectorized NumPy/pandas operations (no Python loops over trade rows), and exposes clean public methods. Metrics include: Sharpe ratio, Sortino ratio, Calmar ratio, recovery factor, profit factor, max drawdown, win rate, avg win/loss, win/loss ratio, expectancy, consecutive streaks, rolling Sharpe, and group aggregations by symbol, direction, hour, weekday, and duration bucket. Each metric raises `InsufficientDataError` with a specific message when there is not enough data to compute it reliably.

### `charts.py`
One function per chart, each returning a `plotly.graph_objects.Figure`. All charts are fully interactive — hover tooltips show P&L, trade count, and win rate; every chart supports zoom, pan, and PNG export. Charts include: equity curve with drawdown shading, daily P&L bars, rolling Sharpe line, P&L distribution histogram, P&L by symbol, long/short breakdown, duration analysis, time of day analysis, day of week analysis, and gross vs net P&L with fee drag shading.

### `report.py`
Assembles the final HTML file from chart figures and metric data. Converts each Plotly figure to an embeddable HTML div, builds the metric card grid, applies the Bloomberg-style CSS, and writes the complete standalone file to disk. Separated from chart logic so either layer can be modified without touching the other.

### `__main__.py`
CLI entry point. Parses and validates arguments before loading any data — raises a clear error if the CSV path does not exist or the account size is non-positive. Supports `--bullets-only` flag to print resume-ready bullet points to stdout without regenerating the full dashboard. Structured logging with timestamps and module names throughout.

---

## Running the tests

```bash
PYTHONPATH=src pytest tests/ -v
```

```
38 passed in 1.90s
```

Tests cover: symbol parsing, duration parsing, formatters, data loading, net P&L calculation, win flag correctness, all scalar metrics, edge cases (single trading day, all-winning days, empty CSV, missing file), and all group aggregations.

---

## Resume output

```bash
PYTHONPATH=src python3 -m trade_journal --csv data/trades.csv --account 50000 --bullets-only
```

```
• Apr 2026 – present live trading track record: 21 trades across MCL, MNQ futures
• Sharpe 11.388, profit factor 3.399, max drawdown 0.01% of account
• Win rate 52.38%, avg win/loss ratio 3.09:1, expectancy $33.40/trade
• Net P&L $701.36 over 4 trading days; all sessions within daily loss limits
```

---

## Metrics reference

| Metric | Formula | Benchmark |
|---|---|---|
| Sharpe | (mean excess daily return / std) × √252 | > 1.0 good, > 1.5 strong |
| Sortino | (mean excess return / downside std) × √252 | > 1.0 good |
| Calmar | annualized return / max drawdown | > 0.5 good |
| Profit factor | gross profit / gross loss | > 1.5 good, > 2.0 strong |
| Expectancy | (win% × avg win) + (loss% × avg loss) | must be > $0 |
| Recovery factor | net P&L / max drawdown | > 3.0 strong |

---

## Stack

Python 3.10+ · pandas · NumPy · Plotly · pytest

---

## Data privacy

Real trade history is excluded via `.gitignore`. The `data/sample_trades.csv` file contains synthetic data matching Topstep's export format for demonstration purposes only.

---

*Track record verified through Topstep funded account. Past performance does not guarantee future results.*
