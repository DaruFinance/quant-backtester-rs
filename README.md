# Quant Backtester (Rust)

Speed-optimized Rust port of the batch strategy runner from the [Quant Research Framework](https://github.com/DaruFinance/quant-research-framework). Generates ~20,000 strategy variants from a combinatorial grid of indicators, transformations, and confluence filters, then evaluates each one with walk-forward optimization and robustness stress tests — all running in parallel across all available cores.

> **Research / educational use only. Not financial advice. Past backtest results do not guarantee future performance.**

This is the performance-critical engine behind the research pipeline. The Python version runs the same logic; this Rust version replaces all O(n * window) rolling computations with O(n) algorithms and caches signal computations across robustness scenarios to avoid redundant work, yielding significant speedups on large datasets.

---

## What It Looks Like

**Startup — loads data, generates strategy grid, runs on all threads:**

```
Loading data from: data/SOLUSDT_1h.csv
Config: BACKTEST_CANDLES=10000, OOS_CANDLES=120000, WFO_TRIGGER_VAL=5000
Loaded 48853 bars in 0.01s
Generated 19806 strategy specs
[runner] Running 19806 pending strategies on 32 threads
[runner] Completed 100/19806 strategies (0.5%)
[runner] Completed 500/19806 strategies (2.5%)
[runner] Completed 1000/19806 strategies (5.0%)
...
[runner] All 19806 strategies completed in 312.4s
[runner] Output root: strategy_outputs
```

**Per-strategy output — reproducibility header, IS/OOS metrics, robustness tests, WFO windows:**

```
=== RUNNING STRATEGY: STOCHK_EMA_55_atr_pct0.8_SL2 ===
Reproducibility Parameters:
  SLIPPAGE_PCT = 0.03
  FEE_PCT = 0.05
  FUNDING_FEE = 0.01
  ...
 OOS-raw | Trades: 235  ROI:$974.26  PF:  1.15  Shp:  1.05  Win: 46.81%  Exp:$4.36  MaxDD:$1,125.68

Best Sharpe look-back = 55

 OOS-opt (LB 55) | Trades: 235  ROI:$974.26  PF:  1.15  Shp:  1.05  Win: 46.81%  Exp:$4.36  MaxDD:$1,125.68

Robustness Test: ENT (Test 1)
 ENT OOS1 | Trades: 235  ROI:$424.82  PF:  1.06  Shp:  0.50  Win: 44.26%  Exp:$2.04  MaxDD:$1,330.00

Robustness Test: FEE (Test 2)
 FEE OOS1 | Trades: 235  ROI:$677.11  PF:  1.10  Shp:  0.78  Win: 46.81%  Exp:$3.18  MaxDD:$1,153.30

Robustness Test: SLI (Test 3)
 SLI OOS1 | Trades: 235  ROI:$534.02  PF:  1.09  Shp:  0.69  Win: 45.53%  Exp:$2.66  MaxDD:$1,298.84
```

---

## Data Source

Feed this tool OHLC candle data in CSV format (`time,open,high,low,close`). The easiest way to get it is with the Binance downloader included in the Python framework:

```bash
# From the companion repo:
python binance_ohlc_downloader.py --symbol SOLUSDT --interval 1h --market spot --source api --since 2020-01-01 --until now --out data/SOLUSDT_1h.csv
```

See: https://github.com/DaruFinance/quant-research-framework

---

## Performance vs Python

The Rust version applies two classes of optimization over the Python original:

**Algorithmic — O(n) rolling computations:**

| Function | Python | Rust |
|----------|--------|------|
| SMA / rolling mean | O(n * w) slice sum | O(n) running sum |
| Rolling std (ddof=0, ddof=1) | O(n * w) two-pass | O(n) Welford running sum + sum-of-squares |
| Rolling min / max | O(n * w) slice scan | O(n) monotonic deque |
| Rolling sum | O(n * w) slice sum | O(n) running accumulator |
| Rolling variance | O(n * w) two-pass | O(n) running sum + sum-of-squares |
| Stochastic K | O(n * w) per-bar min/max | O(n) pre-computed rolling min/max |

**Structural — signal caching in robustness tests:**

The Python version recomputes `create_raw_signals` + `parse_signals` + `apply_confluence_filter` for every robustness scenario, even when the lookback period hasn't changed. The Rust version pre-computes baseline and drifted signals once, then reuses them. Signals are only recomputed when indicator variance is active (which actually changes the lookback).

**Execution — parallel strategy evaluation:**

All ~20,000 strategies run in parallel via `rayon`, saturating all available cores. The Python version runs strategies with `multiprocessing.Pool`.

### Benchmark Results

Identical workload (same 50 strategies, same CSV, same settings) on the same machine (AMD Ryzen 9 7950X, 32 threads):

| | Python (32 workers) | Rust (32 threads) | Speedup |
|---|---|---|---|
| **50 strategies** | 70.2s | 0.5s | **140x** |
| **Per strategy** | 1,404ms | 10ms | **140x** |
| **Projected 19,806** | ~7.7 hours | ~3.3 min | |

> Benchmark methodology: both versions ran the first 50 strategies from the same deterministic generation order, with identical parameters (SOLUSDT 1h, 48,853 bars, 10k IS / 120k OOS candles, WFO trigger 5000, SL 2%). Python used `multiprocessing.Pool` with 32 spawn workers; Rust used `rayon::par_iter` with 32 threads.

---

## Strategy Grid

The engine generates strategies from a combinatorial grid:

| Dimension | Values |
|-----------|--------|
| **Primary indicator** | EMA, SMA, RSI, RSI Level, MACD, StochK, ATR, PPO |
| **Partner periods** | Indicator-specific (e.g. 20/50/100 for EMA, 8/20/50/100 for RSI) |
| **Transformations** (12) | none, zscore, slope, normalized_price, roc, bias, volZ, accel, disFromMedian, quant_stretch, rank_resid, fold_dev |
| **Confluences** (25) | none, RSIge40, RSIge50, Pge0.7, Pge0.8, kurtosis, kurtosis10, skew, skew0.75, atr_pct, atr_pct0.8, BW_filter, pi, vr, burstfreq, TinyBody, NoNewLowGreen, RangeSpike, YesterdayPeak, DeadFlat10, InsideBar, SameDirection, TopOfRange, VolContraction, EMAHug |
| **Transform modes** | calc, src (where applicable) |
| **Stop losses** | Configurable (default: [2.0]) |

Total with default settings: **19,806 strategies**.

Each strategy goes through:
1. In-sample / out-of-sample baseline backtest
2. Lookback optimization (coarse-to-fine grid search with smart optimization)
3. RRR (reward-to-risk ratio) optimization
4. 4 robustness stress tests (entry drift, fee shock, slippage shock, indicator variance)
5. Walk-forward optimization with rolling re-optimization per window

---

## Quick Start

### 1. Build

```bash
cargo build --release
```

### 2. Generate a config file (optional)

```bash
./target/release/quant-backtester-rs --generate-config
```

This writes a `config.toml` with all defaults. Edit it to match your dataset:

```toml
csv_file = "SOLUSDT_1h.csv"
account_size = 100000
risk_amount = 2500
slippage_pct = 0.03
fee_pct = 0.05
funding_fee = 0.01
backtest_candles = 10000
oos_candles = 120000
opt_metric = "Sharpe"
min_trades = 1
use_sl = true
use_tp = true
tp_percentage = 3
optimize_rrr = true
use_wfo = true
wfo_trigger_mode = "candles"
wfo_trigger_val = 5000
stop_loss_values = [2]
```

### 3. Run

```bash
# With config.toml in the current directory:
./target/release/quant-backtester-rs output_dir

# Or override via CLI args:
./target/release/quant-backtester-rs output_dir data/SOLUSDT_1h.csv 10000 120000 5000 2.0

# Binary trade output (smaller files, use convert_trades to expand):
./target/release/quant-backtester-rs output_dir data/SOLUSDT_1h.csv --bin
```

### 4. Analyze results

The output is compatible with the [Strategy Generalization Analysis](https://github.com/DaruFinance/strategy-generalization-analysis) toolkit. Point it at your output directory to run robustness funnels, portfolio simulations, and META sliding-window analysis.

---

## Configuration Reference

All parameters can be set in `config.toml` or overridden via CLI positional args.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_file` | `SOLUSDT_1h.csv` | Path to OHLC CSV file |
| `account_size` | `100000` | Simulated account balance |
| `risk_amount` | `2500` | Position size per trade |
| `slippage_pct` | `0.03` | Slippage as percentage of price |
| `fee_pct` | `0.05` | Trading fee percentage |
| `funding_fee` | `0.01` | Crypto funding fee (applied at 00:00, 08:00, 16:00 UTC) |
| `backtest_candles` | `10000` | In-sample window size (candles) |
| `oos_candles` | `120000` | Out-of-sample window size (candles) |
| `opt_metric` | `Sharpe` | Metric to optimize: Sharpe, ROI, PF, WinRate, Exp, MaxDrawdown |
| `min_trades` | `1` | Minimum trades required for a valid backtest |
| `use_sl` | `true` | Enable stop-loss |
| `use_tp` | `true` | Enable take-profit |
| `tp_percentage` | `3.0` | Take-profit distance (% of entry price) |
| `optimize_rrr` | `true` | Optimize reward-to-risk ratio across candidates [1..5] |
| `use_wfo` | `true` | Enable walk-forward optimization |
| `wfo_trigger_mode` | `candles` | WFO window advancement mode |
| `wfo_trigger_val` | `5000` | Candles per WFO window |
| `stop_loss_values` | `[2.0]` | Stop-loss percentages to test (cross-product with all strategies) |

---

## Output Structure

```
output_dir/
├── EMA/
│   ├── EMA_x_EMA20_SL2/
│   │   ├── EMA_x_EMA20_SL2.txt      # Full metrics log
│   │   ├── trade_list.csv            # Per-trade records (or trades.bin with --bin)
│   │   └── _done                     # Completion marker (enables resume)
│   ├── EMA_x_EMA50_zscore_SL2/
│   │   └── ...
│   └── ...
├── SMA/
├── RSI/
├── MACD/
├── STOCHK/
├── ATR/
└── PPO/
```

The runner is **resumable** — if interrupted, it detects `_done` markers and skips completed strategies on restart.

---

## Architecture

```
main()
  ├── Load OHLC data (CSV → Vec<Bar>)
  ├── generate_all_strategies()        → 19,806 StrategySpec variants
  ├── get_completed()                  → skip already-done strategies
  └── par_iter (rayon)                 → one thread per strategy
        └── run_strategy()
              ├── Baseline IS/OOS backtest (default lookback)
              ├── optimiser()          → coarse-to-fine lookback search
              │     ├── Coarse scan (step=2 across lookback range)
              │     ├── Fine refine (neighbors of best)
              │     ├── Smart optimization (reject PF spikes >10%)
              │     └── RRR optimization (test reward ratios 1..5)
              ├── Robustness tests     → 4 stress scenarios
              │     ├── Entry drift
              │     ├── Fee shock (2x fees)
              │     ├── Slippage shock (3x slippage)
              │     └── Entry drift + indicator variance
              └── Walk-forward windows → rolling re-optimization
```

---

## Related Projects

| Repository | Language | Purpose |
|------------|----------|---------|
| [quant-research-framework](https://github.com/DaruFinance/quant-research-framework) | Python | Original backtester with WFO, robustness tests, and Binance data downloader |
| [strategy-generalization-analysis](https://github.com/DaruFinance/strategy-generalization-analysis) | Python | Post-hoc analysis: robustness funnels, portfolio simulations, META sliding-window |
| **this repo** | Rust | Speed-optimized batch runner — same logic, O(n) algorithms, parallel execution |

---

## Disclaimer

This tool is for **research and educational purposes only**. Nothing in this repository constitutes financial advice. Backtests and simulations do not guarantee future results. Use at your own risk.
