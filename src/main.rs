use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use rand;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Section 1: Configuration
//
// All tunable parameters live in `AppConfig`, which can be loaded from a
// `config.toml` file or populated with sensible defaults.  A hand-rolled TOML
// parser handles basic key = value, string, boolean, integer, float, and
// bracket-delimited arrays — no external crate required.
//
// At runtime the heavy-weight `RunParams` singleton caches values that were
// historically supplied via CLI args so the rest of the code can read them
// through cheap accessor functions.
// ---------------------------------------------------------------------------

struct AppConfig {
    csv_file: String,
    account_size: f64,
    risk_amount: f64,
    slippage_pct: f64,
    fee_pct: f64,
    funding_fee: f64,
    backtest_candles: usize,
    oos_candles: usize,
    opt_metric: String,
    min_trades: usize,
    use_monte_carlo: bool,
    mc_runs: usize,
    use_sl: bool,
    use_tp: bool,
    tp_percentage: f64,
    optimize_rrr: bool,
    use_wfo: bool,
    wfo_trigger_mode: String,
    wfo_trigger_val: usize,
    stop_loss_values: Vec<f64>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            csv_file: "SOLUSDT_1h.csv".into(),
            account_size: 100_000.0,
            risk_amount: 2_500.0,
            slippage_pct: 0.03,
            fee_pct: 0.05,
            funding_fee: 0.01,
            backtest_candles: 10_000,
            oos_candles: 120_000,
            opt_metric: "Sharpe".into(),
            min_trades: 1,
            use_monte_carlo: false,
            mc_runs: 1000,
            use_sl: true,
            use_tp: true,
            tp_percentage: 3.0,
            optimize_rrr: true,
            use_wfo: true,
            wfo_trigger_mode: "candles".into(),
            wfo_trigger_val: 5000,
            stop_loss_values: vec![2.0],
        }
    }
}

impl AppConfig {
    fn generate_sample_toml() -> String {
        let c = Self::default();
        format!(
r#"# Quant Backtester Configuration
csv_file = "{}"
account_size = {}
risk_amount = {}
slippage_pct = {}
fee_pct = {}
funding_fee = {}
backtest_candles = {}
oos_candles = {}
opt_metric = "{}"
min_trades = {}
use_monte_carlo = {}
mc_runs = {}
use_sl = {}
use_tp = {}
tp_percentage = {}
optimize_rrr = {}
use_wfo = {}
wfo_trigger_mode = "{}"
wfo_trigger_val = {}
stop_loss_values = [{}]
"#,
            c.csv_file, c.account_size, c.risk_amount, c.slippage_pct,
            c.fee_pct, c.funding_fee, c.backtest_candles, c.oos_candles,
            c.opt_metric, c.min_trades, c.use_monte_carlo, c.mc_runs,
            c.use_sl, c.use_tp, c.tp_percentage, c.optimize_rrr,
            c.use_wfo, c.wfo_trigger_mode, c.wfo_trigger_val,
            c.stop_loss_values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        )
    }

    fn load_from_toml(path: &str) -> Self {
        let mut cfg = Self::default();
        let contents = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => return cfg,
        };
        for raw_line in contents.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() { continue; }
            let mut parts = line.splitn(2, '=');
            let key = match parts.next() { Some(k) => k.trim(), None => continue };
            let val = match parts.next() { Some(v) => v.trim(), None => continue };
            match key {
                "csv_file" => cfg.csv_file = strip_quotes(val),
                "account_size" => if let Ok(v) = val.parse() { cfg.account_size = v; },
                "risk_amount" => if let Ok(v) = val.parse() { cfg.risk_amount = v; },
                "slippage_pct" => if let Ok(v) = val.parse() { cfg.slippage_pct = v; },
                "fee_pct" => if let Ok(v) = val.parse() { cfg.fee_pct = v; },
                "funding_fee" => if let Ok(v) = val.parse() { cfg.funding_fee = v; },
                "backtest_candles" => if let Ok(v) = val.parse() { cfg.backtest_candles = v; },
                "oos_candles" => if let Ok(v) = val.parse() { cfg.oos_candles = v; },
                "opt_metric" => cfg.opt_metric = strip_quotes(val),
                "min_trades" => if let Ok(v) = val.parse() { cfg.min_trades = v; },
                "use_monte_carlo" => cfg.use_monte_carlo = parse_bool(val),
                "mc_runs" => if let Ok(v) = val.parse() { cfg.mc_runs = v; },
                "use_sl" => cfg.use_sl = parse_bool(val),
                "use_tp" => cfg.use_tp = parse_bool(val),
                "tp_percentage" => if let Ok(v) = val.parse() { cfg.tp_percentage = v; },
                "optimize_rrr" => cfg.optimize_rrr = parse_bool(val),
                "use_wfo" => cfg.use_wfo = parse_bool(val),
                "wfo_trigger_mode" => cfg.wfo_trigger_mode = strip_quotes(val),
                "wfo_trigger_val" => if let Ok(v) = val.parse() { cfg.wfo_trigger_val = v; },
                "stop_loss_values" => cfg.stop_loss_values = parse_f64_array(val),
                _ => {}
            }
        }
        cfg
    }
}

fn strip_quotes(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        s[1..s.len()-1].to_string()
    } else {
        s.to_string()
    }
}

fn parse_bool(s: &str) -> bool {
    matches!(s.trim().to_lowercase().as_str(), "true" | "1" | "yes")
}

fn parse_f64_array(s: &str) -> Vec<f64> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    inner.split(',')
        .filter_map(|v| v.trim().parse::<f64>().ok())
        .collect()
}

static APP_CONFIG: OnceLock<AppConfig> = OnceLock::new();

fn app() -> &'static AppConfig {
    APP_CONFIG.get().expect("AppConfig not initialised")
}

const USE_OOS2: bool = false;
const SMART_OPTIMIZATION: bool = true;
const DRAWDOWN_CONSTRAINT: Option<f64> = None;
const USE_REGIME_SEG: bool = false;
const _FAST_EMA_SPAN: usize = 20;
const AGE_DATASET: usize = 0;
const RRR_CANDIDATES: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];

struct RunParams {
    backtest_candles: usize,
    oos_candles: usize,
    wfo_trigger_val: usize,
    sl_values: Vec<f64>,
}
static RUN_PARAMS: OnceLock<RunParams> = OnceLock::new();

fn backtest_candles() -> usize {
    RUN_PARAMS.get().map_or(app().backtest_candles, |p| p.backtest_candles)
}
fn oos_candles_base() -> usize {
    RUN_PARAMS.get().map_or(app().oos_candles, |p| p.oos_candles)
}
fn wfo_trigger_val() -> usize {
    RUN_PARAMS.get().map_or(app().wfo_trigger_val, |p| p.wfo_trigger_val)
}
fn stop_loss_values() -> Vec<f64> {
    RUN_PARAMS.get().map_or(app().stop_loss_values.clone(), |p| p.sl_values.clone())
}

fn robustness_scenarios() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        ("Test 1", vec!["ENTRY_DRIFT"]),
        ("Test 2", vec!["FEE_SHOCK"]),
        ("Test 3", vec!["SLIPPAGE_SHOCK"]),
        ("Test 4", vec!["ENTRY_DRIFT", "INDICATOR_VARIANCE"]),
    ]
}
const MAX_ROBUSTNESS_SCENARIOS: usize = 5;
const METRICS_LIST: [&str; 6] = ["ROI", "PF", "Sharpe", "WinRate", "Exp", "MaxDrawdown"];

// ---------------------------------------------------------------------------
// Section 2: Core Data Structures
//
// `Bar` is a single OHLC candle.  `Trade` records a completed round-trip.
// `Metrics` holds the standard performance statistics computed after a
// backtest run.  `Config` (the per-strategy config, distinct from AppConfig)
// carries the fee / slippage / position-size parameters that the backtest
// engine reads on every tick.
// ---------------------------------------------------------------------------

fn python_iloc_idx(idx: isize, length: usize) -> usize {
    if idx >= 0 { (idx as usize).min(length) }
    else if (-idx) as usize <= length { (length as isize + idx) as usize }
    else { 0 }
}

fn python_iloc_slice(start_raw: i64, end_raw: i64, length: usize) -> (usize, usize) {
    let s = python_iloc_idx(start_raw as isize, length);
    let e = python_iloc_idx(end_raw as isize, length);
    if s >= e { (s, s) } else { (s, e) }
}

#[derive(Clone)]
struct Bar {
    time_unix: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Clone, Debug)]
struct Trade {
    side: i8,
    entry_idx: i32,
    exit_idx: i32,
    _entry_price: f64,
    _exit_price: f64,
    _qty: f64,
    pnl: f64,
}

#[derive(Clone, Debug)]
struct Metrics {
    trades: usize,
    roi: f64,
    pf: f64,
    win_rate: f64,
    exp: f64,
    sharpe: f64,
    max_drawdown: f64,
    consistency: f64,
    rrr: Option<usize>,
}

impl Default for Metrics {
    fn default() -> Self {
        Metrics {
            trades: 0, roi: 0.0, pf: f64::INFINITY, win_rate: 0.0,
            exp: 0.0, sharpe: 0.0, max_drawdown: 0.0, consistency: 0.0, rrr: None,
        }
    }
}

impl Metrics {
    fn get(&self, key: &str) -> f64 {
        match key {
            "ROI" => self.roi, "PF" => self.pf, "Sharpe" => self.sharpe,
            "WinRate" => self.win_rate, "Exp" => self.exp,
            "MaxDrawdown" => self.max_drawdown, "Consistency" => self.consistency, _ => 0.0,
        }
    }
}

#[derive(Clone)]
struct Config {
    tp_percentage: f64,
    use_tp: bool,
    fee_pct: f64,
    slippage_pct: f64,
    oos_candles: usize,
    position_size: f64,
    sl_percentage: f64,
    default_lb: usize,
    min_trades: usize,
}

impl Config {
    fn new(sl: f64, default_lb: usize) -> Self {
        let a = app();
        let oos = if USE_OOS2 { oos_candles_base() * 2 } else { oos_candles_base() };
        Config {
            tp_percentage: a.tp_percentage, use_tp: a.use_tp,
            fee_pct: a.fee_pct, slippage_pct: a.slippage_pct,
            oos_candles: oos, position_size: a.risk_amount,
            sl_percentage: sl, default_lb, min_trades: a.min_trades,
        }
    }
    fn fee_rate(&self) -> f64 { self.fee_pct / 100.0 }
    fn slip(&self) -> f64 { self.slippage_pct * 0.01 }
    fn funding_rate(&self) -> f64 { app().funding_fee / 100.0 }
    fn dd_constraint(&self) -> Option<f64> { DRAWDOWN_CONSTRAINT.map(|d| d / 100.0) }
    fn lookback_range(&self) -> Vec<usize> {
        let lo = (self.default_lb as f64 * 0.25) as usize;
        let hi = (self.default_lb as f64 * 1.5) as usize + 1;
        (lo..hi).collect()
    }
}

fn utc_hour_minute(unix_ts: i64) -> (u32, u32) {
    let secs_in_day = ((unix_ts % 86400) + 86400) % 86400;
    let hour = (secs_in_day / 3600) as u32;
    let minute = ((secs_in_day % 3600) / 60) as u32;
    (hour, minute)
}

// ---------------------------------------------------------------------------
// Section 3: Strategy Specification
//
// Each strategy is described by a `StrategySpec` — an immutable recipe that
// names which primary indicator to use (EMA, SMA, RSI, MACD, etc.), what
// transformation to apply (z-score, slope, ROC, ...), which confluence
// filter gates the signals, and how large the stop-loss is.  The
// combinatorial grid in Section 12 produces thousands of these specs.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Primary { EMA, SMA, RSI, RsiLevel, MACD, StochK, ATR, PPO }

#[derive(Clone, Copy, PartialEq)]
enum PartnerKind { EMA, SMA }

#[derive(Clone, Copy, PartialEq)]
enum Transformation {
    None, Zscore, Slope, NormalizedPrice, Roc, Bias, VolZ,
    Accel, DisFromMedian, QuantStretch, RankResid, FoldDev,
}

#[derive(Clone, Copy, PartialEq)]
enum TransformMode { Calc, Src }

#[derive(Clone, Copy, PartialEq)]
enum Confluence {
    None, RSIge40, RSIge50, Pge0_7, Pge0_8, Kurtosis, Kurtosis10,
    Skew, Skew0_75, AtrPct, AtrPct0_8, BwFilter, Pi, Vr,
    Burstfreq, TinyBody, NoNewLowGreen, RangeSpike, YesterdayPeak,
    DeadFlat10, InsideBar, SameDirection, TopOfRange, VolContraction, EMAHug,
}

#[derive(Clone)]
struct StrategySpec {
    name: String,
    primary: Primary,
    partner_kind: PartnerKind,
    partner_val: Option<usize>,
    macd_params: Option<(usize, usize)>,
    transformation: Transformation,
    transform_mode: TransformMode,
    confluence: Confluence,
    stop_loss: f64,
    default_lb: usize,
}


// ---------------------------------------------------------------------------
// Section 4: Data Loading
//
// Expects a CSV with columns: unix_timestamp, open, high, low, close
// (header row is skipped).  Rows are sorted by timestamp after loading.
// The `age_dataset` helper trims the most recent N bars — useful for
// simulating "what would the system have seen K bars ago".
// ---------------------------------------------------------------------------

fn load_ohlc(path: &str) -> Vec<Bar> {
    let file = File::open(path).unwrap_or_else(|_| panic!("CSV file not found: {}", path));
    let reader = BufReader::new(file);
    let mut bars: Vec<Bar> = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if i == 0 { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 { continue; }
        let time_unix: i64 = fields[0].trim().parse().expect("bad time");
        let open: f64 = fields[1].trim().parse().expect("bad open");
        let high: f64 = fields[2].trim().parse().expect("bad high");
        let low: f64 = fields[3].trim().parse().expect("bad low");
        let close: f64 = fields[4].trim().parse().expect("bad close");
        bars.push(Bar { time_unix, open, high, low, close });
    }
    bars.sort_by_key(|b| b.time_unix);
    bars
}

fn age_dataset(bars: Vec<Bar>, age: usize) -> Vec<Bar> {
    if age == 0 { return bars; }
    bars[..bars.len() - age].to_vec()
}

// ---------------------------------------------------------------------------
// Section 5: Indicators
//
// All indicators use O(n) streaming algorithms where possible (running sums,
// exponential moving averages, monotonic deques for rolling min/max).  NaN
// propagation mirrors pandas semantics so that results are numerically
// identical to the Python reference implementation.
//
// Key functions:
//   compute_ema       — exponential moving average (pandas ewm, adjust=False)
//   compute_sma       — simple moving average via running sum
//   compute_rsi       — RSI using adjusted EWM (pandas ewm, adjust=True)
//   compute_macd      — MACD line + signal line
//   compute_stoch     — Stochastic %K
//   compute_atr       — Average True Range (adjusted EWM)
//   compute_ppo_master — Price Percentage Oscillator
// ---------------------------------------------------------------------------

fn compute_ema(data: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 { return out; }
    let mut ewm = f64::NAN;
    for i in 0..n {
        if data[i].is_nan() { continue; }
        if ewm.is_nan() { ewm = data[i]; }
        else { ewm = alpha * data[i] + (1.0 - alpha) * ewm; }
        out[i] = ewm;
    }
    out
}

fn compute_sma(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    let mut nan_count = 0usize;
    let mut running_sum = 0.0f64;
    for j in 0..window {
        if data[j].is_nan() { nan_count += 1; } else { running_sum += data[j]; }
    }
    if nan_count == 0 { out[window - 1] = running_sum / window as f64; }
    for i in window..n {
        let old = data[i - window];
        let new = data[i];
        if old.is_nan() { nan_count -= 1; } else { running_sum -= old; }
        if new.is_nan() { nan_count += 1; } else { running_sum += new; }
        if nan_count == 0 { out[i] = running_sum / window as f64; }
    }
    out
}

/// RSI via pandas-compatible adjusted EWM (com = length-1).
fn compute_rsi(close: &[f64], length: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < 2 || length == 0 { return out; }
    let alpha = 1.0 / length as f64;

    let mut gains = vec![0.0f64; n];
    let mut losses = vec![0.0f64; n];
    for i in 1..n {
        let d = close[i] - close[i - 1];
        if d > 0.0 { gains[i] = d; }
        else { losses[i] = -d; }
    }

    let avg_gain = ewm_mean_adjusted(&gains, alpha, length);
    let avg_loss = ewm_mean_adjusted(&losses, alpha, length);

    for i in 0..n {
        let ag = avg_gain[i];
        let al = avg_loss[i];
        if ag.is_nan() || al.is_nan() { continue; }
        if al == 0.0 {
            out[i] = 100.0;
        } else {
            let rs = ag / al;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    out
}

/// EWM mean with adjust=True, matching pandas default for RSI / ATR.
fn ewm_mean_adjusted(data: &[f64], alpha: f64, min_periods: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 { return out; }
    let one_minus_alpha = 1.0 - alpha;
    let mut num = data[0];
    let mut den = 1.0;
    let mut count = 1usize;
    if count >= min_periods { out[0] = num / den; }
    for i in 1..n {
        num = data[i] + one_minus_alpha * num;
        den = 1.0 + one_minus_alpha * den;
        count += 1;
        if count >= min_periods { out[i] = num / den; }
    }
    out
}

fn compute_macd(close: &[f64], fast_len: usize, slow_len: usize, signal_len: usize) -> (Vec<f64>, Vec<f64>) {
    let fast_ema = compute_ema(close, fast_len);
    let slow_ema = compute_ema(close, slow_len);
    let n = close.len();
    let mut macd_line = vec![0.0f64; n];
    for i in 0..n { macd_line[i] = fast_ema[i] - slow_ema[i]; }
    let sig_line = compute_ema(&macd_line, signal_len);
    (macd_line, sig_line)
}

fn compute_stoch(high: &[f64], low: &[f64], close: &[f64], length: usize) -> Vec<f64> {
    let n = close.len();
    if n < length || length == 0 { return vec![f64::NAN; n]; }
    let lo_roll = rolling_min(low, length);
    let hi_roll = rolling_max(high, length);
    let mut out = vec![f64::NAN; n];
    for i in (length - 1)..n {
        if lo_roll[i].is_nan() || hi_roll[i].is_nan() { continue; }
        let denom = hi_roll[i] - lo_roll[i];
        out[i] = if denom != 0.0 { 100.0 * (close[i] - lo_roll[i]) / denom } else { f64::NAN };
    }
    out
}

/// ATR via pandas-compatible adjusted EWM (alpha = 1/length).
fn compute_atr(high: &[f64], low: &[f64], close: &[f64], length: usize) -> Vec<f64> {
    let n = close.len();
    if n == 0 || length == 0 { return vec![f64::NAN; n]; }
    let mut tr = vec![0.0f64; n];
    tr[0] = high[0] - low[0];
    for i in 1..n {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        tr[i] = hl.max(hc).max(lc);
    }
    let alpha = 1.0 / length as f64;
    ewm_mean_adjusted(&tr, alpha, length)
}

/// PPO: rolling min/max per element, then rolling sums for the oscillator.
fn compute_ppo_master(high: &[f64], low: &[f64], close: &[f64], fast_len: usize) -> Vec<f64> {
    let n = close.len();
    let mut out = vec![f64::NAN; n];
    if n < fast_len || fast_len == 0 { return out; }

    let lo = rolling_min(low, fast_len);
    let hi = rolling_max(high, fast_len);

    let mut close_minus_lo = vec![f64::NAN; n];
    let mut hi_minus_close = vec![f64::NAN; n];
    let mut hi_minus_lo = vec![f64::NAN; n];
    for i in 0..n {
        if lo[i].is_nan() || hi[i].is_nan() { continue; }
        close_minus_lo[i] = close[i] - lo[i];
        hi_minus_close[i] = hi[i] - close[i];
        hi_minus_lo[i] = hi[i] - lo[i];
    }

    let sum_up = rolling_sum(&close_minus_lo, fast_len);
    let sum_down = rolling_sum(&hi_minus_close, fast_len);
    let den_master = rolling_sum(&hi_minus_lo, fast_len);

    for i in 0..n {
        if sum_up[i].is_nan() || sum_down[i].is_nan() || den_master[i].is_nan() { continue; }
        let num = sum_up[i] - sum_down[i];
        out[i] = if den_master[i] != 0.0 { num / den_master[i] } else { 0.0 };
    }
    out
}


// ---------------------------------------------------------------------------
// Section 6: Transformations (Signal Processing Layer)
//
// Transformations re-express a raw indicator series into a derived form
// before crossover detection.  Each is a pure function from &[f64] -> Vec<f64>.
// Examples: z-score normalisation, slope (finite difference), rate-of-change,
// volatility z-score, acceleration (second derivative), quantile stretch.
//
// Rolling helpers (sum, std, min, max, median, rank, quantile) all use O(n)
// algorithms — running sums for mean/std, monotonic deques for min/max.
// Median and rank fall back to per-window sorts (O(n*w log w)) since they
// are only used in transformations, not the hot backtest loop.
// ---------------------------------------------------------------------------

fn shift_right(data: &[f64], by: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    for i in by..n { out[i] = data[i - by]; }
    out
}

fn rolling_mean(data: &[f64], window: usize) -> Vec<f64> {
    compute_sma(data, window)
}

fn rolling_sum(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    let mut nan_count = 0usize;
    let mut running = 0.0f64;
    for j in 0..window {
        if data[j].is_nan() { nan_count += 1; } else { running += data[j]; }
    }
    if nan_count == 0 { out[window - 1] = running; }
    for i in window..n {
        let old = data[i - window];
        let new = data[i];
        if old.is_nan() { nan_count -= 1; } else { running -= old; }
        if new.is_nan() { nan_count += 1; } else { running += new; }
        if nan_count == 0 { out[i] = running; }
    }
    out
}

fn rolling_std_ddof0(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    let wf = window as f64;
    let mut nan_count = 0usize;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for j in 0..window {
        if data[j].is_nan() { nan_count += 1; } else { s1 += data[j]; s2 += data[j] * data[j]; }
    }
    if nan_count == 0 {
        let var = (s2 / wf) - (s1 / wf).powi(2);
        out[window - 1] = var.max(0.0).sqrt();
    }
    for i in window..n {
        let old = data[i - window];
        let new = data[i];
        if old.is_nan() { nan_count -= 1; } else { s1 -= old; s2 -= old * old; }
        if new.is_nan() { nan_count += 1; } else { s1 += new; s2 += new * new; }
        if nan_count == 0 {
            let var = (s2 / wf) - (s1 / wf).powi(2);
            out[i] = var.max(0.0).sqrt();
        }
    }
    out
}

fn rolling_min(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    let mut nan_count = 0usize;
    for j in 0..window { if data[j].is_nan() { nan_count += 1; } }
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for j in 0..window {
        if !data[j].is_nan() {
            while let Some(&back) = deque.back() {
                if data[back] >= data[j] { deque.pop_back(); } else { break; }
            }
            deque.push_back(j);
        }
    }
    if nan_count == 0 { if let Some(&front) = deque.front() { out[window - 1] = data[front]; } }
    for i in window..n {
        let old = i - window;
        if data[old].is_nan() { nan_count -= 1; }
        if data[i].is_nan() { nan_count += 1; }
        while let Some(&front) = deque.front() { if front <= old { deque.pop_front(); } else { break; } }
        if !data[i].is_nan() {
            while let Some(&back) = deque.back() {
                if data[back] >= data[i] { deque.pop_back(); } else { break; }
            }
            deque.push_back(i);
        }
        if nan_count == 0 { if let Some(&front) = deque.front() { out[i] = data[front]; } }
    }
    out
}

fn rolling_max(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    let mut nan_count = 0usize;
    for j in 0..window { if data[j].is_nan() { nan_count += 1; } }
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    for j in 0..window {
        if !data[j].is_nan() {
            while let Some(&back) = deque.back() {
                if data[back] <= data[j] { deque.pop_back(); } else { break; }
            }
            deque.push_back(j);
        }
    }
    if nan_count == 0 { if let Some(&front) = deque.front() { out[window - 1] = data[front]; } }
    for i in window..n {
        let old = i - window;
        if data[old].is_nan() { nan_count -= 1; }
        if data[i].is_nan() { nan_count += 1; }
        while let Some(&front) = deque.front() { if front <= old { deque.pop_front(); } else { break; } }
        if !data[i].is_nan() {
            while let Some(&back) = deque.back() {
                if data[back] <= data[i] { deque.pop_back(); } else { break; }
            }
            deque.push_back(i);
        }
        if nan_count == 0 { if let Some(&front) = deque.front() { out[i] = data[front]; } }
    }
    out
}

fn rolling_median(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    for i in (window - 1)..n {
        let slice = &data[i + 1 - window..=i];
        if slice.iter().any(|v| v.is_nan()) { continue; }
        let mut vals: Vec<f64> = slice.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = vals.len() / 2;
        out[i] = if vals.len() % 2 == 0 { (vals[mid - 1] + vals[mid]) / 2.0 } else { vals[mid] };
    }
    out
}

fn rolling_rank(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    for i in (window - 1)..n {
        let slice = &data[i + 1 - window..=i];
        if slice.iter().any(|v| v.is_nan()) { continue; }
        let val = data[i];
        let mut sorted: Vec<f64> = slice.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut rank_sum = 0.0;
        let mut count = 0;
        for (j, &v) in sorted.iter().enumerate() {
            if (v - val).abs() < 1e-15 { rank_sum += (j + 1) as f64; count += 1; }
        }
        out[i] = if count > 0 { rank_sum / count as f64 } else { f64::NAN };
    }
    out
}

fn rolling_quantile(data: &[f64], window: usize, q: f64) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    for i in (window - 1)..n {
        let slice = &data[i + 1 - window..=i];
        if slice.iter().any(|v| v.is_nan()) { continue; }
        let mut vals: Vec<f64> = slice.to_vec();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = q * (vals.len() - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        out[i] = vals[lo] * (1.0 - frac) + vals[hi.min(vals.len() - 1)] * frac;
    }
    out
}

fn f_zscore(src: &[f64], length: usize) -> Vec<f64> {
    let n = src.len();
    let mean = rolling_mean(src, length);
    let std = rolling_std_ddof0(src, length);
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if mean[i].is_nan() || std[i].is_nan() { continue; }
        if std[i] == 0.0 { out[i] = f64::NAN; continue; }
        out[i] = (src[i] - mean[i]) / std[i];
    }
    out
}

fn f_slope(src: &[f64], length: usize) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    if length == 0 {
        return vec![0.0; n];
    }
    for i in length..n {
        out[i] = (src[i] - src[i - length]) / length as f64;
    }
    out
}

fn f_normalized_price(src: &[f64], length: usize) -> Vec<f64> {
    let n = src.len();
    let hi = rolling_max(src, length);
    let lo = rolling_min(src, length);
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if hi[i].is_nan() || lo[i].is_nan() { continue; }
        let rng = hi[i] - lo[i];
        out[i] = if rng != 0.0 { (src[i] - lo[i]) / rng } else { 0.0 };
    }
    out
}

fn f_roc(src: &[f64], length: usize) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    for i in length..n {
        let prev = src[i - length];
        out[i] = if prev != 0.0 { (src[i] - prev) / prev * 100.0 } else { f64::NAN };
    }
    out
}

fn f_bias(src: &[f64], length: usize, smooth: usize) -> Vec<f64> {
    let n = src.len();
    let mut slope = vec![f64::NAN; n];
    for i in length..n {
        slope[i] = (src[i] - src[i - length]) / length as f64;
    }
    let mut out = vec![f64::NAN; n];
    let alpha = 2.0 / (smooth as f64 + 1.0);
    let mut started = false;
    let mut ewm = 0.0f64;
    for i in 0..n {
        if slope[i].is_nan() { continue; }
        if !started { ewm = slope[i]; started = true; out[i] = ewm; continue; }
        ewm = alpha * slope[i] + (1.0 - alpha) * ewm;
        out[i] = ewm;
    }
    out
}

fn f_volz(src: &[f64], length: usize) -> Vec<f64> {
    let vol = rolling_std_ddof0(src, length);
    let mean_vol = rolling_mean(&vol, length);
    let dev_vol = rolling_std_ddof0(&vol, length);
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if vol[i].is_nan() || mean_vol[i].is_nan() || dev_vol[i].is_nan() { continue; }
        if dev_vol[i] == 0.0 { out[i] = f64::NAN; continue; }
        out[i] = (vol[i] - mean_vol[i]) / dev_vol[i];
    }
    out
}

fn f_accel(src: &[f64], length: usize) -> Vec<f64> {
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    let len2 = length * 2;
    for i in len2..n {
        out[i] = (src[i] - 2.0 * src[i - length] + src[i - len2]) / (length * length) as f64;
    }
    out
}

fn f_dis_from_median(src: &[f64], length: usize) -> Vec<f64> {
    if length == 0 { return vec![0.0; src.len()]; }
    let hi = rolling_max(src, length);
    let lo = rolling_min(src, length);
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if hi[i].is_nan() || lo[i].is_nan() { continue; }
        let median = (hi[i] + lo[i]) / 2.0;
        out[i] = src[i] - median;
    }
    out
}

fn f_quant_stretch(src: &[f64], lb: usize) -> Vec<f64> {
    let lo = rolling_quantile(src, lb, 0.1);
    let hi = rolling_quantile(src, lb, 0.9);
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if lo[i].is_nan() || hi[i].is_nan() { continue; }
        let den = hi[i] - lo[i];
        if den == 0.0 || den.is_nan() { out[i] = f64::NAN; continue; }
        out[i] = 2.0 * ((src[i] - lo[i]) / den) - 1.0;
    }
    out
}

fn f_rank_resid(src: &[f64], lb: usize) -> Vec<f64> {
    let r = rolling_rank(src, lb);
    let r_mean = rolling_mean(&r, lb);
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    let denom = (lb as f64 - 1.0).max(1.0);
    for i in 0..n {
        if r[i].is_nan() || r_mean[i].is_nan() { continue; }
        out[i] = (r[i] - r_mean[i]) / denom;
    }
    out
}

fn f_fold_dev(src: &[f64], lb: usize) -> Vec<f64> {
    let med = rolling_median(src, lb);
    let n = src.len();
    let mut out = vec![f64::NAN; n];
    for i in 1..n {
        if med[i].is_nan() { continue; }
        let diff = src[i] - med[i];
        if med[i - 1].is_nan() { continue; }
        let sign = if src[i] > med[i - 1] { 1.0 } else if src[i] < med[i - 1] { -1.0 } else { 0.0 };
        out[i] = diff.abs() * sign;
    }
    out
}

fn apply_transformation(src: &[f64], trans: Transformation, lb: usize) -> Vec<f64> {
    match trans {
        Transformation::None => src.to_vec(),
        Transformation::Zscore => f_zscore(src, lb),
        Transformation::Slope => f_slope(src, lb),
        Transformation::NormalizedPrice => f_normalized_price(src, lb),
        Transformation::Roc => f_roc(src, lb),
        Transformation::Bias => f_bias(src, lb, lb),
        Transformation::VolZ => f_volz(src, lb),
        Transformation::Accel => f_accel(src, lb),
        Transformation::DisFromMedian => f_dis_from_median(src, lb),
        Transformation::QuantStretch => f_quant_stretch(src, lb),
        Transformation::RankResid => f_rank_resid(src, lb),
        Transformation::FoldDev => f_fold_dev(src, lb),
    }
}


// ---------------------------------------------------------------------------
// Section 7: Signal Generation (Crossover / State-Based)
//
// Raw signals are generated by comparing a "fast" and "slow" line derived
// from each indicator.  Two modes exist:
//   crossover_to_raw — emits +1/-1 only on the bar where fast crosses slow
//   state_to_raw     — emits +1/-1 on every bar where fast > slow (or < slow)
//
// `create_raw_signals_for_spec` dispatches to the correct indicator, applies
// the chosen transformation (optionally to the source price or to the
// computed indicator), builds the fast/slow pair, and returns the raw signal
// vector.
// ---------------------------------------------------------------------------

fn crossover_to_raw(fast: &[f64], slow: &[f64]) -> Vec<i8> {
    let n = fast.len();
    let mut raw = vec![0i8; n];
    for i in 1..n {
        let f = fast[i]; let s = slow[i];
        let fp = fast[i - 1]; let sp = slow[i - 1];
        if f.is_nan() || s.is_nan() || fp.is_nan() || sp.is_nan() { continue; }
        if f > s && fp <= sp { raw[i] = 1; }
        else if f < s && fp >= sp { raw[i] = -1; }
    }
    raw
}

fn _state_to_raw(fast: &[f64], slow: &[f64]) -> Vec<i8> {
    let n = fast.len();
    let mut raw = vec![0i8; n];
    for i in 1..n {
        let f = fast[i - 1]; let s = slow[i - 1];
        if f.is_nan() || s.is_nan() { continue; }
        if f > s { raw[i] = 1; }
        else if f < s { raw[i] = -1; }
    }
    raw
}

fn create_raw_signals_for_spec(bars: &[Bar], lb: usize, spec: &StrategySpec) -> Vec<i8> {
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let n = bars.len();
    if n == 0 { return vec![]; }

    match spec.primary {
        Primary::EMA => {
            let src = apply_transformation(&close, spec.transformation, lb);
            let partner_period = spec.partner_val.unwrap_or(50);
            let fast = shift_right(&compute_ema(&src, partner_period), 1);
            let slow = shift_right(&compute_ema(&src, lb), 1);
            crossover_to_raw(&fast, &slow)
        }
        Primary::SMA => {
            let src = apply_transformation(&close, spec.transformation, lb);
            let partner_period = spec.partner_val.unwrap_or(50);
            let fast = shift_right(&compute_sma(&src, partner_period), 1);
            let slow = shift_right(&compute_sma(&src, lb), 1);
            crossover_to_raw(&fast, &slow)
        }
        Primary::RSI => {
            let rsi_full = if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Src {
                let src_price = apply_transformation(&close, spec.transformation, lb);
                compute_rsi(&src_price, lb)
            } else {
                let mut rsi = compute_rsi(&close, lb);
                if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Calc {
                    rsi = apply_transformation(&rsi, spec.transformation, lb);
                }
                rsi
            };

            let (fast, slow) = match spec.partner_kind {
                PartnerKind::EMA => {
                    let f = shift_right(&compute_ema(&rsi_full, lb), 1);
                    let s = shift_right(&compute_ema(&rsi_full, (1.5 * lb as f64) as usize), 1);
                    (f, s)
                }
                PartnerKind::SMA => {
                    let f = shift_right(&compute_sma(&rsi_full, lb), 1);
                    let s = shift_right(&compute_sma(&rsi_full, (1.5 * lb as f64) as usize), 1);
                    (f, s)
                }
            };
            crossover_to_raw(&fast, &slow)
        }
        Primary::RsiLevel => {
            let rsi_full = compute_rsi(&close, lb);
            let level = spec.partner_val.unwrap_or(50) as f64;
            let fast = match spec.partner_kind {
                PartnerKind::EMA => shift_right(&compute_ema(&rsi_full, 3), 1),
                PartnerKind::SMA => shift_right(&compute_sma(&rsi_full, 3), 1),
            };
            let slow = vec![level; n];
            crossover_to_raw(&fast, &slow)
        }
        Primary::MACD => {
            let src = apply_transformation(&close, spec.transformation, lb);
            let (_fast_len, slow_len) = spec.macd_params.unwrap_or((24, 52));
            let (macd_line, sig_line) = compute_macd(&src, lb, slow_len, 9);
            let macd_sm = compute_sma(&macd_line, 10);
            let sig_sm = compute_sma(&sig_line, 10);
            let fast = shift_right(&macd_sm, 1);
            let slow = shift_right(&sig_sm, 1);
            crossover_to_raw(&fast, &slow)
        }
        Primary::StochK => {
            let k_full = if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Src {
                let src_price = apply_transformation(&close, spec.transformation, lb);
                compute_stoch(&high, &low, &src_price, lb)
            } else {
                let mut k = compute_stoch(&high, &low, &close, lb);
                if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Calc {
                    k = apply_transformation(&k, spec.transformation, lb);
                }
                k
            };

            let (fast, slow) = match spec.partner_kind {
                PartnerKind::EMA => {
                    let f = shift_right(&compute_ema(&k_full, lb), 1);
                    let s = shift_right(&compute_ema(&k_full, 2 * lb), 1);
                    (f, s)
                }
                PartnerKind::SMA => {
                    let f = shift_right(&compute_sma(&k_full, lb), 1);
                    let s = shift_right(&compute_sma(&k_full, 2 * lb), 1);
                    (f, s)
                }
            };
            crossover_to_raw(&fast, &slow)
        }
        Primary::ATR => {
            let atr_full = if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Src {
                let price_warped = apply_transformation(&close, spec.transformation, lb);
                compute_atr(&high, &low, &price_warped, lb)
            } else {
                let mut atr = compute_atr(&high, &low, &close, lb);
                if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Calc {
                    atr = apply_transformation(&atr, spec.transformation, lb);
                }
                atr
            };

            let fast = shift_right(&compute_sma(&atr_full, 3), 1);
            let partner_period = spec.partner_val.unwrap_or(50);
            let slow = match spec.partner_kind {
                PartnerKind::EMA => shift_right(&compute_ema(&atr_full, partner_period), 1),
                PartnerKind::SMA => shift_right(&compute_sma(&atr_full, partner_period), 1),
            };
            crossover_to_raw(&fast, &slow)
        }
        Primary::PPO => {
            let ppo_full = if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Src {
                let src_price = apply_transformation(&close, spec.transformation, lb);
                compute_ppo_master(&high, &low, &src_price, lb)
            } else {
                let mut ppo = compute_ppo_master(&high, &low, &close, lb);
                if spec.transformation != Transformation::None && spec.transform_mode == TransformMode::Calc {
                    ppo = apply_transformation(&ppo, spec.transformation, lb);
                }
                ppo
            };

            let fast = shift_right(&ppo_full, 1);
            let half_lb = (0.5 * lb as f64) as usize;
            let slow = match spec.partner_kind {
                PartnerKind::EMA => shift_right(&compute_ema(&ppo_full, half_lb), 1),
                PartnerKind::SMA => shift_right(&compute_sma(&ppo_full, half_lb), 1),
            };
            crossover_to_raw(&fast, &slow)
        }
    }
}


// ---------------------------------------------------------------------------
// Section 8: Confluences (Filtering Layer)
//
// A confluence is an additional boolean gate applied after signal generation.
// It suppresses entries that occur in unfavourable market micro-conditions.
// Examples include RSI threshold filters, price-position-in-range checks,
// kurtosis / skewness regime tests, ATR percentile filters, body-width
// ratios, volume contraction detectors, and candlestick pattern checks.
//
// Each confluence produces a boolean mask; entries where the mask is false
// are zeroed out.
// ---------------------------------------------------------------------------

fn compute_confluence_mask(bars: &[Bar], raw: &[i8], lb: usize, confluence: Confluence) -> Vec<bool> {
    let n = bars.len();
    if confluence == Confluence::None { return vec![true; n]; }

    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();
    let high: Vec<f64> = bars.iter().map(|b| b.high).collect();
    let low: Vec<f64> = bars.iter().map(|b| b.low).collect();
    let open: Vec<f64> = bars.iter().map(|b| b.open).collect();

    match confluence {
        Confluence::None => vec![true; n],

        Confluence::RSIge50 => {
            let rsi = shift_right(&compute_rsi(&close, 14), 1);
            rsi.iter().map(|&v| !v.is_nan() && v >= 50.0).collect()
        }
        Confluence::RSIge40 => {
            let rsi = shift_right(&compute_rsi(&close, 14), 1);
            rsi.iter().map(|&v| !v.is_nan() && v >= 40.0).collect()
        }
        Confluence::Pge0_8 => {
            let lo_s1 = shift_right(&rolling_min(&close, lb), 1);
            let hi_s1 = shift_right(&rolling_max(&close, lb), 1);
            let mut p_unshifted = vec![f64::NAN; n];
            for i in 0..n {
                if lo_s1[i].is_nan() || hi_s1[i].is_nan() { continue; }
                let denom = hi_s1[i] - lo_s1[i];
                if denom == 0.0 { continue; }
                p_unshifted[i] = (close[i] - lo_s1[i]) / denom;
            }
            let p = shift_right(&p_unshifted, 1);
            let mut keep = vec![false; n];
            for i in 0..n {
                if p[i].is_nan() { continue; }
                keep[i] = (raw[i] == 1 && p[i] >= 0.8) || (raw[i] == -1 && p[i] <= 0.2);
            }
            keep
        }
        Confluence::Pge0_7 => {
            let lo_s1 = shift_right(&rolling_min(&close, lb), 1);
            let hi_s1 = shift_right(&rolling_max(&close, lb), 1);
            let mut p_unshifted = vec![f64::NAN; n];
            for i in 0..n {
                if lo_s1[i].is_nan() || hi_s1[i].is_nan() { continue; }
                let denom = hi_s1[i] - lo_s1[i];
                if denom == 0.0 { continue; }
                p_unshifted[i] = (close[i] - lo_s1[i]) / denom;
            }
            let p = shift_right(&p_unshifted, 1);
            let mut keep = vec![false; n];
            for i in 0..n {
                if p[i].is_nan() { continue; }
                keep[i] = (raw[i] == 1 && p[i] >= 0.7) || (raw[i] == -1 && p[i] <= 0.3);
            }
            keep
        }
        Confluence::BwFilter => {
            let mut keep = vec![true; n];
            for i in 1..n {
                let denom = high[i - 1] - low[i - 1];
                if denom == 0.0 { continue; }
                let bw = (close[i - 1] - open[i - 1]).abs() / denom;
                if raw[i] == 1 && bw <= 0.7 { keep[i] = false; }
            }
            keep
        }
        Confluence::Pi => {
            let atr = compute_atr(&high, &low, &close, lb);
            let mut keep = vec![false; n];
            let pi_mid = std::f64::consts::PI / 4.0;
            let tol = 0.05;
            for i in 1..n {
                if atr[i - 1].is_nan() || atr[i - 1] == 0.0 { continue; }
                let body = (close[i - 1] - open[i - 1]).abs();
                let ratio = body / atr[i - 1];
                keep[i] = ratio >= (pi_mid - tol) && ratio <= (pi_mid + tol);
            }
            keep
        }
        Confluence::Vr => {
            let mut ret = vec![f64::NAN; n];
            for i in 1..n { ret[i] = close[i] - close[i - 1]; }
            let sigma1 = rolling_var_ddof0(&ret, lb);
            let mut r2 = vec![f64::NAN; n];
            for i in 2..n { r2[i] = close[i] - close[i - 2]; }
            let sigma2 = rolling_var_ddof0(&r2, lb);
            let mut keep = vec![false; n];
            for i in 1..n {
                if sigma1[i - 1].is_nan() || sigma2[i - 1].is_nan() || sigma1[i - 1] == 0.0 { continue; }
                let vr = sigma2[i - 1] / (2.0 * sigma1[i - 1]);
                keep[i] = vr >= 1.0;
            }
            keep
        }
        Confluence::Kurtosis => { kurtosis_mask(&close, lb, 5.0) }
        Confluence::Kurtosis10 => { kurtosis_mask(&close, lb, 10.0) }
        Confluence::Skew => { skew_mask(&close, lb, 0.5) }
        Confluence::Skew0_75 => { skew_mask(&close, lb, 0.75) }
        Confluence::AtrPct => { atr_pct_mask(&high, &low, &close, lb, 0.7) }
        Confluence::AtrPct0_8 => { atr_pct_mask(&high, &low, &close, lb, 0.8) }
        Confluence::Burstfreq => {
            let mut ret = vec![f64::NAN; n];
            for i in 1..n { ret[i] = close[i] - close[i - 1]; }
            let sd = rolling_std_ddof1(&ret, lb);
            let mut bursts = vec![0.0f64; n];
            for i in 0..n {
                if !sd[i].is_nan() && !ret[i].is_nan() && ret[i].abs() > sd[i] { bursts[i] = 1.0; }
            }
            let freq = shift_right(&rolling_mean(&bursts, lb), 1);
            freq.iter().map(|&v| !v.is_nan() && v >= 0.2).collect()
        }
        Confluence::TinyBody => {
            let mut keep = vec![false; n];
            for i in 1..n {
                let body = (close[i - 1] - open[i - 1]).abs();
                let range = high[i - 1] - low[i - 1];
                keep[i] = body < range * 0.1;
            }
            keep
        }
        Confluence::NoNewLowGreen => {
            let mut keep = vec![false; n];
            for i in 6..n {
                let green = close[i - 1] > open[i - 1];
                let no_new_low = low[i - 1] > close[i - 6];
                keep[i] = green && no_new_low;
            }
            keep
        }
        Confluence::RangeSpike => {
            let mut range_vec = vec![0.0f64; n];
            for i in 0..n { range_vec[i] = high[i] - low[i]; }
            let avg_rng = rolling_mean(&range_vec, 20);
            let mut keep = vec![false; n];
            for i in 1..n {
                if avg_rng[i - 1].is_nan() { continue; }
                keep[i] = range_vec[i - 1] > 1.3 * avg_rng[i - 1];
            }
            keep
        }
        Confluence::YesterdayPeak => {
            let close_max3 = rolling_max(&close, 3);
            let mut keep = vec![false; n];
            for i in 2..n {
                if close_max3[i - 1].is_nan() { continue; }
                keep[i] = (close[i - 2] - close_max3[i - 1]).abs() < 1e-12;
            }
            keep
        }
        Confluence::DeadFlat10 => {
            let mut keep = vec![false; n];
            for i in 11..n {
                let ret = (close[i - 1] / close[i - 11] - 1.0).abs();
                keep[i] = ret < 0.005;
            }
            keep
        }
        Confluence::InsideBar => {
            let mut keep = vec![false; n];
            for i in 2..n {
                keep[i] = high[i - 2] <= high[i - 1] && low[i - 2] >= low[i - 1];
            }
            keep
        }
        Confluence::SameDirection => {
            let mut keep = vec![false; n];
            if n >= 2 {
                keep[1] = !(close[0] > open[0]);
            }
            for i in 2..n {
                let curr_up = close[i - 1] > open[i - 1];
                let prev_up = close[i - 2] > open[i - 2];
                keep[i] = curr_up == prev_up;
            }
            keep
        }
        Confluence::TopOfRange => {
            let highest = rolling_max(&high, 50);
            let mut keep = vec![false; n];
            for i in 1..n {
                if highest[i - 1].is_nan() { continue; }
                keep[i] = close[i - 1] > highest[i - 1] * 0.99;
            }
            keep
        }
        Confluence::VolContraction => {
            let mut rng = vec![0.0f64; n];
            for i in 0..n { rng[i] = high[i] - low[i]; }
            let short_std = rolling_std_ddof1(&rng, 10);
            let long_std = rolling_std_ddof1(&rng, 100);
            let mut keep = vec![false; n];
            for i in 1..n {
                if short_std[i - 1].is_nan() || long_std[i - 1].is_nan() { continue; }
                keep[i] = short_std[i - 1] < long_std[i - 1];
            }
            keep
        }
        Confluence::EMAHug => {
            let ema21 = rolling_mean(&close, 21);
            let mut keep = vec![false; n];
            for i in 1..n {
                if ema21[i - 1].is_nan() { continue; }
                let dev = (close[i - 1] - ema21[i - 1]).abs();
                let range = high[i - 1] - low[i - 1];
                keep[i] = dev < range * 0.92;
            }
            keep
        }
    }
}

fn rolling_var_ddof0(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window == 0 || n < window { return out; }
    let wf = window as f64;
    let mut nan_count = 0usize;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for j in 0..window {
        if data[j].is_nan() { nan_count += 1; } else { s1 += data[j]; s2 += data[j] * data[j]; }
    }
    if nan_count == 0 { out[window - 1] = ((s2 / wf) - (s1 / wf).powi(2)).max(0.0); }
    for i in window..n {
        let old = data[i - window];
        let new = data[i];
        if old.is_nan() { nan_count -= 1; } else { s1 -= old; s2 -= old * old; }
        if new.is_nan() { nan_count += 1; } else { s1 += new; s2 += new * new; }
        if nan_count == 0 { out[i] = ((s2 / wf) - (s1 / wf).powi(2)).max(0.0); }
    }
    out
}

fn kurtosis_mask(close: &[f64], lb: usize, threshold: f64) -> Vec<bool> {
    let n = close.len();
    let mut r = vec![f64::NAN; n];
    for i in 1..n { r[i] = close[i] / close[i - 1] - 1.0; }
    let mu = rolling_mean(&r, lb);
    let mut dev = vec![0.0f64; n];
    for i in 0..n {
        if r[i].is_nan() || mu[i].is_nan() { dev[i] = f64::NAN; }
        else { dev[i] = r[i] - mu[i]; }
    }
    let mut m2_data = vec![f64::NAN; n];
    let mut m4_data = vec![f64::NAN; n];
    for i in 0..n {
        if dev[i].is_nan() { continue; }
        m2_data[i] = dev[i].powi(2);
        m4_data[i] = dev[i].powi(4);
    }
    let m2 = rolling_mean(&m2_data, lb);
    let m4 = rolling_mean(&m4_data, lb);
    let mut keep = vec![false; n];
    for i in 1..n {
        if m2[i - 1].is_nan() || m4[i - 1].is_nan() || m2[i - 1] == 0.0 { continue; }
        let kappa = m4[i - 1] / (m2[i - 1] * m2[i - 1]);
        keep[i] = kappa >= threshold;
    }
    keep
}

fn skew_mask(close: &[f64], lb: usize, threshold: f64) -> Vec<bool> {
    let n = close.len();
    let mut r = vec![f64::NAN; n];
    for i in 1..n { r[i] = close[i] / close[i - 1] - 1.0; }

    let mu = rolling_mean(&r, lb);
    let sd = rolling_std_ddof1(&r, lb);

    let mut z = vec![f64::NAN; n];
    for i in 0..n {
        if r[i].is_nan() || mu[i].is_nan() || sd[i].is_nan() || sd[i] == 0.0 { continue; }
        z[i] = (r[i] - mu[i]) / sd[i];
    }

    let mut z3 = vec![f64::NAN; n];
    for i in 0..n {
        if z[i].is_nan() { continue; }
        z3[i] = z[i].powi(3);
    }
    let m3 = rolling_mean(&z3, lb);

    let mut keep = vec![false; n];
    for i in 1..n {
        if m3[i - 1].is_nan() { continue; }
        keep[i] = m3[i - 1].abs() >= threshold;
    }
    keep
}

fn rolling_std_ddof1(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![f64::NAN; n];
    if window <= 1 || n < window { return out; }
    let wf = window as f64;
    let wm1 = (window - 1) as f64;
    let mut nan_count = 0usize;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for j in 0..window {
        if data[j].is_nan() { nan_count += 1; } else { s1 += data[j]; s2 += data[j] * data[j]; }
    }
    if nan_count == 0 {
        let var = (s2 - s1 * s1 / wf) / wm1;
        out[window - 1] = var.max(0.0).sqrt();
    }
    for i in window..n {
        let old = data[i - window];
        let new = data[i];
        if old.is_nan() { nan_count -= 1; } else { s1 -= old; s2 -= old * old; }
        if new.is_nan() { nan_count += 1; } else { s1 += new; s2 += new * new; }
        if nan_count == 0 {
            let var = (s2 - s1 * s1 / wf) / wm1;
            out[i] = var.max(0.0).sqrt();
        }
    }
    out
}

fn atr_pct_mask(high: &[f64], low: &[f64], close: &[f64], lb: usize, threshold: f64) -> Vec<bool> {
    let atr = compute_atr(high, low, close, lb);
    let n = atr.len();
    let lo = shift_right(&rolling_min(&atr, lb), 1);
    let hi = shift_right(&rolling_max(&atr, lb), 1);
    let atr_shifted = shift_right(&atr, 1);
    let mut keep = vec![false; n];
    for i in 0..n {
        if lo[i].is_nan() || hi[i].is_nan() || atr_shifted[i].is_nan() { continue; }
        let denom = hi[i] - lo[i];
        if denom == 0.0 { continue; }
        let pct = (atr_shifted[i] - lo[i]) / denom;
        keep[i] = pct >= threshold;
    }
    keep
}


// ---------------------------------------------------------------------------
// Section 9: Signal Parsing and Confluence Application
//
// `parse_signals` converts the raw +1/-1 crossover stream into the signal
// codes used by the backtest engine (1 = go long, 3 = go short).
// `apply_confluence_filter` zeros out entries that fail the confluence mask.
// ---------------------------------------------------------------------------

fn parse_signals(raw: &[i8]) -> Vec<i8> {
    let n = raw.len();
    let mut sig = vec![0i8; n];
    let mut pos: i8 = 0;
    let mut in_prev = true;
    for i in 0..n {
        let r = raw[i];
        if !in_prev { pos = r; in_prev = true; continue; }
        if r == 1 && pos != 1 { sig[i] = 1; pos = 1; }
        else if r == -1 && pos != -1 { sig[i] = 3; pos = -1; }
    }
    sig
}

fn apply_confluence_filter(sig: &mut [i8], bars: &[Bar], raw: &[i8], lb: usize, confluence: Confluence) {
    if confluence == Confluence::None { return; }
    let keep = compute_confluence_mask(bars, raw, lb, confluence);
    for i in 0..sig.len() {
        if (sig[i] == 1 || sig[i] == 3) && !keep[i] {
            sig[i] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Section 10: Backtest Core (Event-Driven Simulation)
//
// The backtest engine walks through bars sequentially, maintaining a single
// open position at a time.  On each bar it:
//   1. Applies funding fees at 0h/8h/16h UTC (crypto perpetual convention).
//   2. Checks stop-loss and take-profit against the bar's high/low.
//   3. Processes entry/exit signal codes (1=long, 3=short, 2=close-long,
//      4=close-short).
//   4. Records PnL per trade after slippage and fees.
//
// Metrics (ROI, profit factor, Sharpe, win rate, expectancy, max drawdown,
// consistency) are computed from the resulting trade list and equity curve.
// ---------------------------------------------------------------------------

fn backtest_core(bars: &[Bar], sig: &[i8], cfg: &Config) -> (Vec<Trade>, Metrics, Vec<f64>, Vec<f64>) {
    let a = app();
    let n = bars.len();
    let fee_rate = cfg.fee_rate();
    let slip = cfg.slip();
    let funding_rate = cfg.funding_rate();
    let position_size = cfg.position_size;
    let sl_perc = cfg.sl_percentage;
    let tp_perc = cfg.tp_percentage;

    let funding_mask: Vec<bool> = bars.iter().map(|b| {
        let (h, m) = utc_hour_minute(b.time_unix);
        m == 0 && (h == 0 || h == 8 || h == 16)
    }).collect();

    let mut trades: Vec<Trade> = Vec::new();
    let mut equity_list: Vec<f64> = vec![a.account_size];
    let mut funding_acc = 0.0f64;
    let mut open_pos: i8 = 0;
    let mut ent_bar: i32 = -1;
    let mut entry_price = 0.0f64;
    let mut qty = 0.0f64;
    let mut fee_entry = 0.0f64;

    for idx in 0..n {
        if open_pos != 0 && funding_mask[idx] {
            let fee_f = qty * bars[idx].open * funding_rate;
            funding_acc += fee_f;
            let last = equity_list.len() - 1;
            equity_list[last] -= fee_f;
        }
        let code = sig[idx];
        if USE_REGIME_SEG && idx < 200 { continue; }
        let price_open = bars[idx].open;

        if open_pos != 0 && code != 1 && code != 3 {
            let sl_pr = if open_pos == 1 { entry_price * (1.0 - sl_perc/100.0) }
                        else { entry_price * (1.0 + sl_perc/100.0) };
            let tp_pr = if open_pos == 1 { entry_price * (1.0 + tp_perc/100.0) }
                        else { entry_price * (1.0 - tp_perc/100.0) };
            let hit_sl = if open_pos == 1 { bars[idx].low <= sl_pr } else { bars[idx].high >= sl_pr };
            let mut hit_tp = if open_pos == 1 { bars[idx].high >= tp_pr } else { bars[idx].low <= tp_pr };
            if hit_sl && hit_tp { hit_tp = false; }
            let is_sl_hit = if a.use_sl && hit_sl { Some(true) }
                            else if cfg.use_tp && hit_tp { Some(false) }
                            else { None };
            if let Some(sl_hit) = is_sl_hit {
                let raw_exit = if sl_hit { sl_pr } else { tp_pr };
                let exit_price = if open_pos == 1 { raw_exit * (1.0 - slip) }
                                 else { raw_exit * (1.0 + slip) };
                let fee_exit = qty * exit_price * fee_rate;
                let pnl = if open_pos == 1 {
                    qty * (exit_price - entry_price) - (fee_entry + fee_exit + funding_acc)
                } else {
                    qty * (entry_price - exit_price) - (fee_entry + fee_exit + funding_acc)
                };
                funding_acc = 0.0;
                trades.push(Trade { side: open_pos, entry_idx: ent_bar, exit_idx: idx as i32,
                    _entry_price: entry_price, _exit_price: exit_price, _qty: qty, pnl });
                let last_eq = *equity_list.last().unwrap();
                equity_list.push(last_eq + pnl);
                open_pos = 0;
                continue;
            }
        }

        if code == 1 {
            if open_pos == -1 {
                let exit_price = price_open * (1.0 + slip);
                let fee_exit = qty * exit_price * fee_rate;
                let pnl = qty * (entry_price - exit_price) - (fee_entry + fee_exit + funding_acc);
                funding_acc = 0.0;
                trades.push(Trade { side: -1, entry_idx: ent_bar, exit_idx: idx as i32,
                    _entry_price: entry_price, _exit_price: exit_price, _qty: qty, pnl });
                let last_eq = *equity_list.last().unwrap();
                equity_list.push(last_eq + pnl);
                open_pos = 0;
            }
            if open_pos == 0 {
                fee_entry = position_size * fee_rate;
                entry_price = price_open * (1.0 + slip);
                qty = position_size / entry_price;
                open_pos = 1; ent_bar = idx as i32;
            }
        } else if code == 3 {
            if open_pos == 1 {
                let exit_price = price_open * (1.0 - slip);
                let fee_exit = qty * exit_price * fee_rate;
                let pnl = qty * (exit_price - entry_price) - (fee_entry + fee_exit + funding_acc);
                funding_acc = 0.0;
                trades.push(Trade { side: 1, entry_idx: ent_bar, exit_idx: idx as i32,
                    _entry_price: entry_price, _exit_price: exit_price, _qty: qty, pnl });
                let last_eq = *equity_list.last().unwrap();
                equity_list.push(last_eq + pnl);
                open_pos = 0;
            }
            if open_pos == 0 {
                fee_entry = position_size * fee_rate;
                entry_price = price_open * (1.0 - slip);
                qty = position_size / entry_price;
                open_pos = -1; ent_bar = idx as i32;
            }
        } else if code == 2 && open_pos == 1 {
            let exit_price = price_open * (1.0 - slip);
            let fee_exit = qty * exit_price * fee_rate;
            let pnl = qty * (exit_price - entry_price) - (fee_entry + fee_exit + funding_acc);
            funding_acc = 0.0;
            trades.push(Trade { side: 1, entry_idx: ent_bar, exit_idx: idx as i32,
                _entry_price: entry_price, _exit_price: exit_price, _qty: qty, pnl });
            let last_eq = *equity_list.last().unwrap();
            equity_list.push(last_eq + pnl);
            open_pos = 0;
        } else if code == 4 && open_pos == -1 {
            let exit_price = price_open * (1.0 + slip);
            let fee_exit = qty * exit_price * fee_rate;
            let pnl = qty * (entry_price - exit_price) - (fee_entry + fee_exit + funding_acc);
            funding_acc = 0.0;
            trades.push(Trade { side: -1, entry_idx: ent_bar, exit_idx: idx as i32,
                _entry_price: entry_price, _exit_price: exit_price, _qty: qty, pnl });
            let last_eq = *equity_list.last().unwrap();
            equity_list.push(last_eq + pnl);
            open_pos = 0;
        }
    }

    if open_pos != 0 {
        let price_last = bars[n - 1].open;
        let exit_price = if open_pos == 1 { price_last * (1.0 - slip) }
                         else { price_last * (1.0 + slip) };
        let fee_exit = qty * exit_price * fee_rate;
        let pnl = if open_pos == 1 {
            qty * (exit_price - entry_price) - (fee_entry + fee_exit + funding_acc)
        } else {
            qty * (entry_price - exit_price) - (fee_entry + fee_exit + funding_acc)
        };
        trades.push(Trade { side: open_pos, entry_idx: ent_bar, exit_idx: (n-1) as i32,
            _entry_price: entry_price, _exit_price: exit_price, _qty: qty, pnl });
        let last_eq = *equity_list.last().unwrap();
        equity_list.push(last_eq + pnl);
    }

    let eq_frac: Vec<f64> = equity_list.iter().map(|e| e / a.account_size).collect();
    let rets: Vec<f64> = trades.iter().map(|t| t.pnl / a.account_size).collect();
    let metrics = compute_metrics(&rets, &eq_frac);
    (trades, metrics, eq_frac, rets)
}

fn run_backtest(bars: &[Bar], sig: &[i8], cfg: &Config) -> (Vec<Trade>, Metrics, Vec<f64>, Vec<f64>) {
    backtest_core(bars, sig, cfg)
}

fn compute_metrics(rets: &[f64], eq_frac: &[f64]) -> Metrics {
    let tc = rets.len();
    if tc == 0 {
        let mut m = Metrics::default();
        m.pf = f64::INFINITY;
        return m;
    }
    let wr = rets.iter().filter(|&&r| r > 0.0).count() as f64 / tc as f64;
    let roi = eq_frac.last().unwrap() - 1.0;
    let wins_sum: f64 = rets.iter().filter(|&&r| r > 0.0).sum();
    let losses_sum: f64 = rets.iter().filter(|&&r| r <= 0.0).map(|r| -r).sum();
    let pf = if losses_sum > 0.0 { wins_sum / losses_sum } else { f64::INFINITY };
    let wins_count = rets.iter().filter(|&&r| r > 0.0).count();
    let losses_count = rets.iter().filter(|&&r| r <= 0.0).count();
    let mw = if wins_count > 0 { wins_sum / wins_count as f64 } else { 0.0 };
    let ml = if losses_count > 0 { losses_sum / losses_count as f64 } else { 0.0 };
    let exp = mw * wr - ml * (1.0 - wr);
    let mean: f64 = rets.iter().sum::<f64>() / tc as f64;
    let variance: f64 = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / tc as f64;
    let std = variance.sqrt();
    let shp = if tc > 1 && std > 0.0 { mean / std * (tc as f64).sqrt() } else { 0.0 };
    let mut hw = vec![0.0f64; eq_frac.len()];
    hw[0] = eq_frac[0];
    for i in 1..eq_frac.len() { hw[i] = hw[i-1].max(eq_frac[i]); }
    let dd = (0..eq_frac.len()).map(|i| if hw[i] > 0.0 { (hw[i]-eq_frac[i])/hw[i] } else { 0.0 }).fold(0.0f64, f64::max);
    let w = [0.0117, 0.0317, 0.0861, 0.2341, 0.6364];
    let segments = split_into_5(rets);
    let seg_sums: Vec<f64> = segments.iter().map(|s| s.iter().sum::<f64>()).collect();
    let weighted: f64 = w.iter().zip(seg_sums.iter()).map(|(wi, si)| wi * si).sum();
    let consistency = 0.6 * weighted + 0.4 * roi;
    Metrics { trades: tc, roi, pf, win_rate: wr, exp, sharpe: shp, max_drawdown: dd, consistency, rrr: None }
}

fn split_into_5(arr: &[f64]) -> Vec<Vec<f64>> {
    let n = arr.len();
    let mut result = Vec::with_capacity(5);
    let mut start = 0;
    for k in 0..5usize {
        let end = start + (n + k) / 5;
        result.push(arr[start..end].to_vec());
        start = end;
    }
    result
}


// ---------------------------------------------------------------------------
// Section 11: Optimizer (Coarse-to-Fine Search with Smart Optimization)
//
// The optimizer searches for the best lookback period by:
//   1. Evaluating every other lookback in the allowed range (coarse pass).
//   2. Refining around the best coarse result by checking its immediate
//      neighbours (fine pass).
//   3. (Smart Optimization) Rejecting candidates whose profit factor spikes
//      more than 10% above their neighbours — a sign of curve-fitting.
//
// When `optimize_rrr` is enabled, each lookback evaluation also sweeps over
// reward-to-risk ratios (1x-5x SL) to find the best take-profit level,
// replicating the Python backtester's RRR search.
// ---------------------------------------------------------------------------

fn optimiser(bars: &[Bar], cfg: &mut Config, spec: &StrategySpec) -> (Option<usize>, Metrics, Option<String>) {
    let a = app();
    let all_lbs = cfg.lookback_range();
    let mut eval_cache: HashMap<usize, Option<(f64, usize, Metrics)>> = HashMap::new();
    let close: Vec<f64> = bars.iter().map(|b| b.close).collect();

    let evaluate = |lb: usize, cfg: &mut Config, cache: &mut HashMap<usize, Option<(f64, usize, Metrics)>>| -> Option<(f64, usize, Metrics)> {
        if let Some(cached) = cache.get(&lb) { return cached.clone(); }
        let raw = create_raw_signals_for_spec(bars, lb, spec);
        let mut sig = parse_signals(&raw);
        apply_confluence_filter(&mut sig, bars, &raw, lb, spec.confluence);
        let met;
        if !a.optimize_rrr {
            let (_, m, _, _) = run_backtest(bars, &sig, cfg);
            met = m;
        } else {
            let old_tp = cfg.tp_percentage;
            let old_use = cfg.use_tp;
            cfg.tp_percentage = 5.0 * cfg.sl_percentage;
            cfg.use_tp = true;
            let (probe_trades, _, _, _) = run_backtest(bars, &sig, cfg);
            let mut peak_rs: Vec<f64> = Vec::new();
            let mut close_rs_vec: Vec<f64> = Vec::new();
            for t in &probe_trades {
                let e = t.entry_idx as usize;
                let x = t.exit_idx as usize;
                if e >= close.len() || x >= close.len() { continue; }
                let ep = close[e];
                let risk = ep * cfg.sl_percentage / 100.0;
                if risk == 0.0 { continue; }
                let trough = bars[e..=x].iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
                peak_rs.push(((ep - trough) / risk).min(3.0));
                close_rs_vec.push((ep - close[x]) / risk);
            }
            let mut best_rrr_val = RRR_CANDIDATES[0];
            let mut best_sum = f64::NEG_INFINITY;
            for &r_target in RRR_CANDIDATES {
                let sum: f64 = peak_rs.iter().zip(close_rs_vec.iter())
                    .map(|(&p, &c)| if p >= r_target { r_target } else { c }).sum();
                if sum > best_sum { best_sum = sum; best_rrr_val = r_target; }
            }
            cfg.tp_percentage = best_rrr_val * cfg.sl_percentage;
            let (_, mut m, _, _) = run_backtest(bars, &sig, cfg);
            m.rrr = Some(best_rrr_val as usize);
            cfg.tp_percentage = old_tp;
            cfg.use_tp = old_use;
            met = m;
        }
        if met.trades < cfg.min_trades { cache.insert(lb, None); return None; }
        if let Some(dd_c) = cfg.dd_constraint() {
            if met.max_drawdown > dd_c { cache.insert(lb, None); return None; }
        }
        let val = if a.opt_metric == "MaxDrawdown" { -met.get(&a.opt_metric) } else { met.get(&a.opt_metric) };
        let result = Some((val, lb, met));
        cache.insert(lb, result.clone());
        result
    };

    let coarse_lbs: Vec<usize> = all_lbs.iter().step_by(2).copied().collect();
    let mut coarse_results: Vec<(f64, usize, Metrics)> = Vec::new();
    for &lb in &coarse_lbs {
        if let Some(r) = evaluate(lb, cfg, &mut eval_cache) { coarse_results.push(r); }
    }
    if coarse_results.is_empty() {
        let default_lb = cfg.default_lb;
        let raw = create_raw_signals_for_spec(bars, default_lb, spec);
        let mut sig = parse_signals(&raw);
        apply_confluence_filter(&mut sig, bars, &raw, default_lb, spec.confluence);
        let (_, m, _, _) = run_backtest(bars, &sig, cfg);
        return (Some(default_lb), m, None);
    }
    coarse_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (_, best_lb, _) = coarse_results[0].clone();
    let idx_in_all = all_lbs.iter().position(|&l| l == best_lb).unwrap();
    let mut candidates: Vec<(f64, usize, Metrics)> = vec![coarse_results[0].clone()];
    if idx_in_all > 0 {
        if let Some(r) = evaluate(all_lbs[idx_in_all - 1], cfg, &mut eval_cache) { candidates.push(r); }
    }
    if idx_in_all + 1 < all_lbs.len() {
        if let Some(r) = evaluate(all_lbs[idx_in_all + 1], cfg, &mut eval_cache) { candidates.push(r); }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut selected = candidates[0].clone();
    let mut smart_msg: Option<String> = None;
    if SMART_OPTIMIZATION {
        let all_lb_set: HashSet<usize> = all_lbs.iter().copied().collect();
        for cand in &candidates {
            let (_, lb_cand, ref met_cand) = *cand;
            let pf_cand = met_cand.pf;
            let mut ok = true;
            for &delta in &[-1i64, 1i64] {
                let neigh = (lb_cand as i64 + delta) as usize;
                if all_lb_set.contains(&neigh) {
                    if let Some(neigh_res) = evaluate(neigh, cfg, &mut eval_cache) {
                        if pf_cand > 1.10 * neigh_res.2.pf { ok = false; break; }
                    }
                }
            }
            if ok {
                selected = cand.clone();
                if lb_cand != best_lb {
                    smart_msg = Some(format!("Smart Optimization: switched from LB {} to LB {} because PF spike exceeded 10% vs neighbors.", best_lb, lb_cand));
                }
                break;
            }
        }
    }
    (Some(selected.1), selected.2, smart_msg)
}


// ---------------------------------------------------------------------------
// Section 12: Utilities — Formatting, Robustness, Trade Export
//
// Helper functions for pretty-printing metrics, formatting currency values,
// drifting entry signals (robustness test), and writing trade logs in both
// CSV and compact binary formats.
// ---------------------------------------------------------------------------

fn fmt_ratio(val: f64) -> String {
    if val.is_nan() { format!("{:>6}", "nan") }
    else { format!("{:6.3}", val) }
}

fn fmt_money(val: f64) -> String {
    let s = format!("{:.2}", val.abs());
    let parts: Vec<&str> = s.split('.').collect();
    let int_part = parts[0];
    let dec_part = parts[1];
    let chars: Vec<char> = int_part.chars().collect();
    let mut result = String::new();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 { result.push(','); }
        result.push(*c);
    }
    if val < 0.0 { format!("-{}.{}", result, dec_part) } else { format!("{}.{}", result, dec_part) }
}

fn prettyprint_to(out: &mut String, tag: &str, m: &Metrics, lb: Option<usize>) {
    let a = app();
    let lb_note = if let Some(l) = lb { format!("(LB {}) ", l) } else { String::new() };
    let rrr_note = if let Some(r) = m.rrr { format!("  RRR:{}", r) } else { String::new() };
    out.push_str(&format!("{:>8} {}| Trades:{:4}  ROI:${}  PF:{:6.2}  Shp:{:6.2}  Win:{:6.2}%  Exp:${}  MaxDD:${}{}\n",
        tag, lb_note, m.trades, fmt_money(m.roi * a.account_size), m.pf, m.sharpe,
        m.win_rate * 100.0, fmt_money(m.exp * a.account_size), fmt_money(m.max_drawdown * a.account_size), rrr_note));
}

fn drift_entries(sig: &[i8]) -> Vec<i8> {
    let mut out = vec![0i8; sig.len()];
    for (i, &code) in sig.iter().enumerate() {
        if code == 1 || code == 3 { if i + 1 < sig.len() { out[i + 1] = code; } }
        else if code == 2 || code == 4 { out[i] = code; }
    }
    out
}

#[derive(Clone)]
struct RobustnessOpts { fee_mult: f64, slip_mult: f64, drift_on: bool, var_on: bool }

fn opts_from_flags(flags: &[&str]) -> RobustnessOpts {
    let tokens: Vec<String> = flags.iter().map(|f| f.trim().to_lowercase().replace(' ', "_")).collect();
    RobustnessOpts {
        fee_mult: if tokens.iter().any(|t| t == "fee_shock") { 2.0 } else { 1.0 },
        slip_mult: if tokens.iter().any(|t| t == "slippage_shock") { 3.0 } else { 1.0 },
        drift_on: tokens.iter().any(|t| t == "entry_drift"),
        var_on: tokens.iter().any(|t| t == "indicator_variance"),
    }
}

fn label_from_flags(flags: &[&str]) -> String {
    let parts: Vec<&str> = flags.iter().map(|f| match f.trim().to_lowercase().replace(' ', "_").as_str() {
        "fee_shock" => "FEE", "slippage_shock" => "SLI", "entry_drift" => "ENT",
        "indicator_variance" => "IND", _ => "???",
    }).collect();
    if parts.is_empty() { "NONE".to_string() } else { parts.join("+") }
}

fn export_trades_csv(trades: &[Trade], bars: &[Bar], strat: &str, window: &str, sample: &str,
    writer: &mut BufWriter<File>) {
    for t in trades {
        let ei = t.entry_idx as usize; let xi = t.exit_idx as usize;
        let side_str = if t.side == 1 { "long" } else { "short" };
        writeln!(writer, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            strat, window, sample, side_str,
            bars[ei].time_unix, bars[ei].open, bars[ei].high, bars[ei].low, bars[ei].close,
            bars[xi].time_unix, bars[xi].open, bars[xi].high, bars[xi].low, bars[xi].close,
            t.pnl).unwrap();
    }
}

fn export_trades_bin(trades: &[Trade], strat: &str, window: &str, sample: &str,
    writer: &mut BufWriter<File>) {
    fn write_str(w: &mut impl IoWrite, s: &str) {
        w.write_all(&(s.len() as u16).to_le_bytes()).unwrap();
        w.write_all(s.as_bytes()).unwrap();
    }
    write_str(writer, strat);
    write_str(writer, window);
    write_str(writer, sample);
    writer.write_all(&(trades.len() as u32).to_le_bytes()).unwrap();
    for t in trades {
        writer.write_all(&t.entry_idx.to_le_bytes()).unwrap();
        writer.write_all(&t.exit_idx.to_le_bytes()).unwrap();
        writer.write_all(&[t.side as u8]).unwrap();
        writer.write_all(&t.pnl.to_le_bytes()).unwrap();
    }
}


// ---------------------------------------------------------------------------
// Section 13: Walk-Forward Optimization (WFO)
//
// WFO slides a fixed-size in-sample window across the dataset, re-optimizing
// the lookback period at each step, then evaluating on the immediately
// following out-of-sample segment.  The trigger mode ("candles") determines
// the OOS segment width.  Robustness overlays (fee shock, slippage shock,
// entry drift, indicator variance) are run on every window so degradation
// under stress is visible per-window.
// ---------------------------------------------------------------------------

fn run_strategy(all_bars: &[Bar], spec: &StrategySpec, output_dir: &str, use_bin: bool) -> String {
    let a = app();
    let mut out = String::new();
    let mut cfg = Config::new(spec.stop_loss, spec.default_lb);

    let name = &spec.name;
    let group = match spec.primary {
        Primary::EMA => "EMA", Primary::SMA => "SMA", Primary::RSI => "RSI",
        Primary::RsiLevel => "RSI_LEVEL", Primary::MACD => "MACD",
        Primary::StochK => "STOCHK", Primary::ATR => "ATR", Primary::PPO => "PPO",
    };
    let folder = format!("{}/{}/{}", output_dir, group, name);
    let _ = fs::create_dir_all(&folder);

    let export_path = if use_bin {
        format!("{}/trades.bin", folder)
    } else {
        format!("{}/trade_list.csv", folder)
    };
    let _ = fs::remove_file(&export_path);
    let mut trade_writer = BufWriter::new(File::create(&export_path).expect("Cannot create trade file"));
    if !use_bin {
        writeln!(trade_writer, "strategy,window,sample,side,entry_time,open_entry,high_entry,low_entry,close_entry,exit_time,open_exit,high_exit,low_exit,close_exit,pnl").unwrap();
    }

    out.push_str(&format!("\n=== RUNNING STRATEGY: {} ===\n", name));
    out.push_str("Reproducibility Parameters:\n");
    out.push_str(&format!("  SLIPPAGE_PCT = {}\n", a.slippage_pct));
    out.push_str(&format!("  FEE_PCT = {}\n", a.fee_pct));
    out.push_str(&format!("  FUNDING_FEE = {}\n", a.funding_fee));
    out.push_str("  TRADE_SESSIONS = False\n");
    out.push_str("  SESSION_START = 8:00\n");
    out.push_str("  SESSION_END = 16:50\n");
    out.push_str(&format!("  backtest_candles() = {}\n", backtest_candles()));
    out.push_str(&format!("  OOS_CANDLES = {}\n", oos_candles_base()));
    out.push_str(&format!("  USE_OOS2 = {}\n", if USE_OOS2 { "True" } else { "False" }));
    out.push_str(&format!("  OPT_METRIC = {}\n", a.opt_metric));
    out.push_str(&format!("  MIN_TRADES = {}\n", a.min_trades));
    out.push_str("  PRINT_EQUITY_CURVE = True\n");
    out.push_str(&format!("  USE_MONTE_CARLO = {}\n", if a.use_monte_carlo { "True" } else { "False" }));
    out.push_str(&format!("  MC_RUNS = {}\n", a.mc_runs));
    out.push_str(&format!("  USE_SL = {}\n", if a.use_sl { "True" } else { "False" }));
    out.push_str(&format!("  SL_PERCENTAGE = {:.1}\n", cfg.sl_percentage));
    out.push_str(&format!("  USE_TP = {}\n", if cfg.use_tp { "True" } else { "False" }));
    out.push_str(&format!("  TP_PERCENTAGE = {:.1}\n", cfg.tp_percentage));
    out.push_str(&format!("  OPTIMIZE_RRR = {}\n", if a.optimize_rrr { "True" } else { "False" }));
    out.push_str("  FOREX_MODE = False\n");
    out.push_str("  FILTER_REGIMES = False\n");
    out.push_str("  FILTER_DIRECTIONS = False\n");
    out.push_str(&format!("  USE_REGIME_SEG = {}\n", if USE_REGIME_SEG { "True" } else { "False" }));
    out.push_str("  NEWS_AVOIDER = False\n");
    out.push_str("  FEE_SHOCK = False\n");
    out.push_str("  SLIPPAGE_SHOCK = False\n");
    out.push_str("  NEWS_CANDLES_INJECTION = False\n");
    out.push_str("  ENTRY_DRIFT = True\n");
    out.push_str("  INDICATOR_VARIANCE = True\n");
    out.push_str(&format!("  USE_WFO = {}\n", if a.use_wfo { "True" } else { "False" }));
    out.push_str(&format!("  WFO_TRIGGER_MODE = {}\n", a.wfo_trigger_mode));
    out.push_str(&format!("  wfo_trigger_val() = {}\n", wfo_trigger_val()));

    let n = all_bars.len();
    let oos_candles = cfg.oos_candles;
    let oos_start = python_iloc_idx(n as isize - oos_candles as isize, n);
    let is_start = python_iloc_idx(n as isize - oos_candles as isize - backtest_candles() as isize, n);
    let is_bars = &all_bars[is_start..oos_start];
    let oos_bars = &all_bars[oos_start..n];

    let default_lb = cfg.default_lb;
    let raw_is = create_raw_signals_for_spec(is_bars, default_lb, spec);
    let mut sig_is = parse_signals(&raw_is);
    apply_confluence_filter(&mut sig_is, is_bars, &raw_is, default_lb, spec.confluence);
    let (_, met_is_raw, eq_is_raw, _) = run_backtest(is_bars, &sig_is, &cfg);
    prettyprint_to(&mut out, "IS-raw", &met_is_raw, None);

    let raw_oos = create_raw_signals_for_spec(oos_bars, default_lb, spec);
    let mut sig_oos = parse_signals(&raw_oos);
    apply_confluence_filter(&mut sig_oos, oos_bars, &raw_oos, default_lb, spec.confluence);
    let (_, met_oos_raw, _, _) = run_backtest(oos_bars, &sig_oos, &cfg);
    prettyprint_to(&mut out, "OOS-raw", &met_oos_raw, None);

    out.push_str("\n\u{2728} Replication BEFORE optimisation \u{2728}\n");
    for mm in &METRICS_LIST {
        let is_val = met_is_raw.get(mm);
        let oos_val = met_oos_raw.get(mm);
        let r = if is_val != 0.0 { oos_val / is_val } else { f64::NAN };
        out.push_str(&format!("  {:>12}: {}\n", mm, fmt_ratio(r)));
    }

    let (best_lb, met_is_opt, _smart_msg) = optimiser(is_bars, &mut cfg, spec);

    if let Some(lb) = best_lb {
        let best_rrr = if a.optimize_rrr { met_is_opt.rrr } else { None };
        let rrr_note = if let Some(r) = best_rrr { format!("  |  Best RRR = {}", r) } else { String::new() };
        out.push_str(&format!("\nBest {} look-back = {}{}\n\n", a.opt_metric, lb, rrr_note));
        prettyprint_to(&mut out, "IS-opt", &met_is_opt, Some(lb));

        let old_tp = cfg.tp_percentage; let old_use = cfg.use_tp;
        if let Some(r) = best_rrr { cfg.tp_percentage = r as f64 * cfg.sl_percentage; cfg.use_tp = true; }

        let raw_is_opt = create_raw_signals_for_spec(is_bars, lb, spec);
        let mut sig_is_opt = parse_signals(&raw_is_opt);
        apply_confluence_filter(&mut sig_is_opt, is_bars, &raw_is_opt, lb, spec.confluence);
        let (tr_is_opt, met_is_opt2, _, _rets_is_opt) = run_backtest(is_bars, &sig_is_opt, &cfg);

        let raw_oos_opt = create_raw_signals_for_spec(oos_bars, lb, spec);
        let mut sig_oos_opt = parse_signals(&raw_oos_opt);
        apply_confluence_filter(&mut sig_oos_opt, oos_bars, &raw_oos_opt, lb, spec.confluence);
        let (tr_oos_opt, mut met_oos_opt, _, _) = run_backtest(oos_bars, &sig_oos_opt, &cfg);
        if let Some(r) = best_rrr { met_oos_opt.rrr = Some(r); }

        if use_bin {
            export_trades_bin(&tr_is_opt, name, &format!("LB{}", lb), "IS-opt", &mut trade_writer);
        } else {
            export_trades_csv(&tr_is_opt, is_bars, name, &format!("LB{}", lb), "IS-opt", &mut trade_writer);
        }
        prettyprint_to(&mut out, "OOS-opt", &met_oos_opt, Some(lb));
        if use_bin {
            export_trades_bin(&tr_oos_opt, name, &format!("LB{}", lb), "OOS-opt", &mut trade_writer);
        } else {
            export_trades_csv(&tr_oos_opt, oos_bars, name, &format!("LB{}", lb), "OOS-opt", &mut trade_writer);
        }

        cfg.tp_percentage = old_tp; cfg.use_tp = old_use;

        out.push_str("\n\u{2728} Replication OOS-opt / IS-opt \u{2728}\n");
        for mm in &METRICS_LIST {
            let is_val = met_is_opt2.get(mm);
            let oos_val = met_oos_opt.get(mm);
            let r = if is_val != 0.0 { oos_val / is_val } else { f64::NAN };
            out.push_str(&format!("  {:>12}: {}\n", mm, fmt_ratio(r)));
        }

        out.push_str("\u{2728} Baseline Optimized Metrics \u{2728}\n");
        prettyprint_to(&mut out, "Baseline IS", &met_is_opt2, None);
        prettyprint_to(&mut out, "Baseline OOS", &met_oos_opt, None);

        run_robustness_to(&mut out, all_bars, Some(lb), best_rrr, &cfg, spec);

        if a.use_wfo {
            out.push_str("\u{2728} Running Walk-Forward Windows \u{2728}\n");
            walk_forward_to(&mut out, all_bars, &eq_is_raw, &mut cfg, spec, &mut trade_writer, use_bin);
        }
    }

    let log_path = format!("{}/{}.txt", folder, name);
    if let Ok(mut f) = File::create(&log_path) {
        let _ = f.write_all(out.as_bytes());
    }

    let done_path = format!("{}/_done", folder);
    if let Ok(mut f) = File::create(&done_path) { let _ = f.write_all(b"ok"); }

    out
}

fn run_robustness_to(out: &mut String, all_bars: &[Bar], best_lb: Option<usize>, best_rrr: Option<usize>, cfg: &Config, spec: &StrategySpec) {
    let scenarios = robustness_scenarios();
    let lb = best_lb.unwrap_or(cfg.default_lb);
    let n = all_bars.len();
    let oos_candles = cfg.oos_candles;
    let oos_start = python_iloc_idx(n as isize - oos_candles as isize, n);
    let is_start = python_iloc_idx(n as isize - oos_candles as isize - backtest_candles() as isize, n);
    let is_bars = &all_bars[is_start..oos_start];
    let oos_bars = &all_bars[oos_start..n];

    let raw_is_base = create_raw_signals_for_spec(is_bars, lb, spec);
    let mut sig_is_base = parse_signals(&raw_is_base);
    apply_confluence_filter(&mut sig_is_base, is_bars, &raw_is_base, lb, spec.confluence);
    let raw_oos_base = create_raw_signals_for_spec(oos_bars, lb, spec);
    let mut sig_oos_base = parse_signals(&raw_oos_base);
    apply_confluence_filter(&mut sig_oos_base, oos_bars, &raw_oos_base, lb, spec.confluence);
    let sig_is_drifted = drift_entries(&sig_is_base);
    let sig_oos_drifted = drift_entries(&sig_oos_base);

    for (name, flags) in scenarios.iter().take(MAX_ROBUSTNESS_SCENARIOS) {
        let opts = opts_from_flags(flags);
        if opts.fee_mult == 1.0 && opts.slip_mult == 1.0 && !opts.drift_on && !opts.var_on { continue; }
        let label = label_from_flags(flags);
        out.push_str(&format!("\n\u{2728} Robustness Test: {} ({}) \u{2728}\n", label, name));
        let mut cfg_rb = cfg.clone();
        cfg_rb.fee_pct *= opts.fee_mult;
        cfg_rb.slippage_pct *= opts.slip_mult;
        if let Some(r) = best_rrr { cfg_rb.tp_percentage = r as f64 * cfg_rb.sl_percentage; cfg_rb.use_tp = true; }

        if opts.var_on {
            let offset: i32 = if rand::random::<bool>() { 1 } else { -1 };
            let lb_use = (lb as i32 + offset).max(1) as usize;
            let raw_is = create_raw_signals_for_spec(is_bars, lb_use, spec);
            let mut sig_is = parse_signals(&raw_is);
            apply_confluence_filter(&mut sig_is, is_bars, &raw_is, lb_use, spec.confluence);
            if opts.drift_on { sig_is = drift_entries(&sig_is); }
            let (_, met_is, _, _) = run_backtest(is_bars, &sig_is, &cfg_rb);

            let raw_oos = create_raw_signals_for_spec(oos_bars, lb_use, spec);
            let mut sig_oos = parse_signals(&raw_oos);
            apply_confluence_filter(&mut sig_oos, oos_bars, &raw_oos, lb_use, spec.confluence);
            if opts.drift_on { sig_oos = drift_entries(&sig_oos); }
            let (_, met_oos, _, _) = run_backtest(oos_bars, &sig_oos, &cfg_rb);

            prettyprint_to(out, &format!("{} IS", label), &met_is, None);
            prettyprint_to(out, &format!("{} OOS1", label), &met_oos, None);
        } else {
            let sig_is = if opts.drift_on { &sig_is_drifted } else { &sig_is_base };
            let sig_oos = if opts.drift_on { &sig_oos_drifted } else { &sig_oos_base };
            let (_, met_is, _, _) = run_backtest(is_bars, sig_is, &cfg_rb);
            let (_, met_oos, _, _) = run_backtest(oos_bars, sig_oos, &cfg_rb);

            prettyprint_to(out, &format!("{} IS", label), &met_is, None);
            prettyprint_to(out, &format!("{} OOS1", label), &met_oos, None);
        }
    }
}

fn walk_forward_to(out: &mut String, all_bars: &[Bar], eq_is_baseline: &[f64], cfg: &mut Config, spec: &StrategySpec, trade_writer: &mut BufWriter<File>, use_bin: bool) {
    let a = app();
    let scenarios = robustness_scenarios();
    let items: Vec<_> = scenarios.iter().take(MAX_ROBUSTNESS_SCENARIOS).collect();
    let mut rb_scenarios_parsed: Vec<(String, RobustnessOpts)> = Vec::new();
    for (_name, flags) in &items {
        let opts = opts_from_flags(flags);
        if opts.fee_mult != 1.0 || opts.slip_mult != 1.0 || opts.drift_on || opts.var_on {
            rb_scenarios_parsed.push((label_from_flags(flags), opts));
        }
    }

    let n = all_bars.len();
    let ni = n as i64;
    let oos_candles = cfg.oos_candles as i64;
    let start_total: i64 = ni - oos_candles;
    let mut cur_start: i64 = start_total;
    let mut window_no = 1usize;
    let mut all_oos_rets: Vec<f64> = Vec::new();
    let mut eq_is_first: Option<Vec<f64>> = None;

    while cur_start < ni {
        let cur_end: i64 = if a.wfo_trigger_mode == "candles" {
            (cur_start + wfo_trigger_val() as i64).min(ni)
        } else { ni };

        let is_raw_start = cur_start - backtest_candles() as i64;
        let (is_s, is_e) = python_iloc_slice(is_raw_start, cur_start, n);
        let (oos_s, oos_e) = python_iloc_slice(cur_start, cur_end, n);
        let is_bars_roll = &all_bars[is_s..is_e];
        let (lb_roll, _, smart_msg) = optimiser(is_bars_roll, cfg, spec);
        if let Some(ref msg) = smart_msg { out.push_str(&format!("{}\n", msg)); }
        if lb_roll.is_none() { break; }
        let lb = lb_roll.unwrap();
        let oos_slice = &all_bars[oos_s..oos_e];

        let (rets_oos, eq_is_window) = run_wfo_window_to(
            out, is_bars_roll, oos_slice, lb, &format!("W{:02}", window_no),
            cfg, &rb_scenarios_parsed, spec, trade_writer, use_bin);

        if eq_is_first.is_none() { eq_is_first = Some(eq_is_window); }
        all_oos_rets.extend_from_slice(&rets_oos);
        cur_start = cur_end;
        window_no += 1;
    }

    let eq_seed = eq_is_first.as_deref().unwrap_or(eq_is_baseline);
    let seed_last = *eq_seed.last().unwrap_or(&1.0);
    let cum_oos: f64 = all_oos_rets.iter().sum();
    out.push_str("\n WFO Summary \n");
    out.push_str(&format!("  Total OOS return segments: {}\n", all_oos_rets.len()));
    out.push_str(&format!("  Total OOS ROI: ${:.2}\n", cum_oos * a.account_size));
    out.push_str(&format!("  Final equity: ${:.2}\n", (seed_last + cum_oos) * a.account_size));
}

fn run_wfo_window_to(out: &mut String, is_bars: &[Bar], oos_bars: &[Bar], lb: usize, window_tag: &str,
    cfg: &Config, rb_scenarios: &[(String, RobustnessOpts)], spec: &StrategySpec,
    trade_writer: &mut BufWriter<File>, use_bin: bool,
) -> (Vec<f64>, Vec<f64>) {
    let name = &spec.name;
    let raw_is = create_raw_signals_for_spec(is_bars, lb, spec);
    let mut sig_is = parse_signals(&raw_is);
    apply_confluence_filter(&mut sig_is, is_bars, &raw_is, lb, spec.confluence);
    let (tr_is, met_is, eq_is, _) = run_backtest(is_bars, &sig_is, cfg);

    let raw_oos = create_raw_signals_for_spec(oos_bars, lb, spec);
    let mut sig_oos = parse_signals(&raw_oos);
    apply_confluence_filter(&mut sig_oos, oos_bars, &raw_oos, lb, spec.confluence);
    let (tr_oos, met_oos, _, rets_oos) = run_backtest(oos_bars, &sig_oos, cfg);

    prettyprint_to(out, &format!("{} IS", window_tag), &met_is, Some(lb));
    prettyprint_to(out, &format!("{} OOS", window_tag), &met_oos, Some(lb));

    if use_bin {
        export_trades_bin(&tr_is, name, &format!("LB{}", lb), &format!("{}-IS", window_tag), trade_writer);
        export_trades_bin(&tr_oos, name, &format!("LB{}", lb), &format!("{}-OOS", window_tag), trade_writer);
    } else {
        export_trades_csv(&tr_is, is_bars, name, &format!("LB{}", lb), &format!("{}-IS", window_tag), trade_writer);
        export_trades_csv(&tr_oos, oos_bars, name, &format!("LB{}", lb), &format!("{}-OOS", window_tag), trade_writer);
    }

    let sig_is_drifted = drift_entries(&sig_is);
    let sig_oos_drifted = drift_entries(&sig_oos);

    for (label, opts) in rb_scenarios {
        if opts.fee_mult == 1.0 && opts.slip_mult == 1.0 && !opts.drift_on && !opts.var_on { continue; }
        let mut cfg_rb = cfg.clone();
        cfg_rb.fee_pct *= opts.fee_mult;
        cfg_rb.slippage_pct *= opts.slip_mult;

        if opts.var_on {
            let offset: i32 = if rand::random::<bool>() { 1 } else { -1 };
            let lb_rb = (lb as i32 + offset).max(1) as usize;

            let raw_is_rb = create_raw_signals_for_spec(is_bars, lb_rb, spec);
            let mut sig_is_rb = parse_signals(&raw_is_rb);
            apply_confluence_filter(&mut sig_is_rb, is_bars, &raw_is_rb, lb_rb, spec.confluence);
            if opts.drift_on { sig_is_rb = drift_entries(&sig_is_rb); }
            let (_, met_is_rb, _, _) = run_backtest(is_bars, &sig_is_rb, &cfg_rb);

            let raw_oos_rb = create_raw_signals_for_spec(oos_bars, lb_rb, spec);
            let mut sig_oos_rb = parse_signals(&raw_oos_rb);
            apply_confluence_filter(&mut sig_oos_rb, oos_bars, &raw_oos_rb, lb_rb, spec.confluence);
            if opts.drift_on { sig_oos_rb = drift_entries(&sig_oos_rb); }
            let (_, met_oos_rb, _, _) = run_backtest(oos_bars, &sig_oos_rb, &cfg_rb);

            prettyprint_to(out, &format!("{} IS+{}", window_tag, label), &met_is_rb, Some(lb_rb));
            prettyprint_to(out, &format!("{} OOS+{}", window_tag, label), &met_oos_rb, Some(lb_rb));
        } else {
            let sig_is_use = if opts.drift_on { &sig_is_drifted } else { &sig_is };
            let sig_oos_use = if opts.drift_on { &sig_oos_drifted } else { &sig_oos };
            let (_, met_is_rb, _, _) = run_backtest(is_bars, sig_is_use, &cfg_rb);
            let (_, met_oos_rb, _, _) = run_backtest(oos_bars, sig_oos_use, &cfg_rb);

            prettyprint_to(out, &format!("{} IS+{}", window_tag, label), &met_is_rb, Some(lb));
            prettyprint_to(out, &format!("{} OOS+{}", window_tag, label), &met_oos_rb, Some(lb));
        }
    }

    (rets_oos, eq_is)
}


// ---------------------------------------------------------------------------
// Section 14: Strategy Generation (Combinatorial Grid)
//
// `generate_all_strategies` builds the full cross-product of:
//   primary indicator  x  partner periods  x  transformations  x
//   transform modes    x  confluences      x  stop-loss values
//
// This produces tens of thousands of `StrategySpec` instances that are then
// run in parallel.  The naming convention encodes every axis so each strategy
// is uniquely identifiable from its name alone.
// ---------------------------------------------------------------------------

fn confluence_from_str(s: &str) -> Confluence {
    match s {
        "RSIge40" => Confluence::RSIge40, "RSIge50" => Confluence::RSIge50,
        "Pge0.7" => Confluence::Pge0_7, "Pge0.8" => Confluence::Pge0_8,
        "kurtosis10" => Confluence::Kurtosis10, "kurtosis" => Confluence::Kurtosis,
        "skew0.75" => Confluence::Skew0_75, "skew" => Confluence::Skew,
        "atr_pct0.8" => Confluence::AtrPct0_8, "atr_pct" => Confluence::AtrPct,
        "BW_filter" => Confluence::BwFilter, "pi" => Confluence::Pi, "vr" => Confluence::Vr,
        "burstfreq" => Confluence::Burstfreq, "TinyBody" => Confluence::TinyBody,
        "NoNewLowGreen" => Confluence::NoNewLowGreen, "RangeSpike" => Confluence::RangeSpike,
        "YesterdayPeak" => Confluence::YesterdayPeak, "DeadFlat10" => Confluence::DeadFlat10,
        "InsideBar" => Confluence::InsideBar, "SameDirection" => Confluence::SameDirection,
        "TopOfRange" => Confluence::TopOfRange, "VolContraction" => Confluence::VolContraction,
        "EMAHug" => Confluence::EMAHug,
        _ => Confluence::None,
    }
}

fn transformation_from_str(s: &str) -> Transformation {
    match s {
        "zscore" => Transformation::Zscore, "slope" => Transformation::Slope,
        "normalized_price" => Transformation::NormalizedPrice, "roc" => Transformation::Roc,
        "bias" => Transformation::Bias, "volZ" => Transformation::VolZ,
        "accel" => Transformation::Accel, "disFromMedian" => Transformation::DisFromMedian,
        "quant_stretch" => Transformation::QuantStretch, "rank_resid" => Transformation::RankResid,
        "fold_dev" => Transformation::FoldDev,
        _ => Transformation::None,
    }
}

const TRANSFORMATIONS: &[&str] = &[
    "", "zscore", "slope", "normalized_price", "roc", "bias", "volZ",
    "accel", "disFromMedian", "quant_stretch", "rank_resid", "fold_dev"
];

const CONFLUENCES: &[&str] = &[
    "", "RSIge40", "Pge0.7", "kurtosis10", "skew0.75", "atr_pct0.8",
    "RSIge50", "Pge0.8", "BW_filter", "pi", "vr", "kurtosis", "skew",
    "atr_pct", "burstfreq", "TinyBody", "NoNewLowGreen", "RangeSpike",
    "YesterdayPeak", "DeadFlat10", "InsideBar", "SameDirection",
    "TopOfRange", "VolContraction", "EMAHug"
];

fn generate_all_strategies() -> Vec<StrategySpec> {
    let mut specs: Vec<StrategySpec> = Vec::new();

    for &p in &[20usize, 50, 100] {
        for &trans in TRANSFORMATIONS {
            for &conf in CONFLUENCES {
                let trans_suffix = if trans.is_empty() { String::new() } else { format!("_{}", trans) };
                let conf_suffix = if conf.is_empty() { String::new() } else { format!("_{}", conf) };
                let name = format!("EMA_x_EMA{}{}{}", p, trans_suffix, conf_suffix);
                specs.push(StrategySpec {
                    name, primary: Primary::EMA, partner_kind: PartnerKind::EMA,
                    partner_val: Some(p), macd_params: None,
                    transformation: transformation_from_str(trans),
                    transform_mode: TransformMode::Calc,
                    confluence: confluence_from_str(conf),
                    stop_loss: 0.0, default_lb: p + 20,
                });
            }
        }
    }

    for &p in &[20usize, 50, 100] {
        for &trans in TRANSFORMATIONS {
            for &conf in CONFLUENCES {
                let trans_suffix = if trans.is_empty() { String::new() } else { format!("_{}", trans) };
                let conf_suffix = if conf.is_empty() { String::new() } else { format!("_{}", conf) };
                let name = format!("SMA_x_SMA{}{}{}", p, trans_suffix, conf_suffix);
                specs.push(StrategySpec {
                    name, primary: Primary::SMA, partner_kind: PartnerKind::SMA,
                    partner_val: Some(p), macd_params: None,
                    transformation: transformation_from_str(trans),
                    transform_mode: TransformMode::Calc,
                    confluence: confluence_from_str(conf),
                    stop_loss: 0.0, default_lb: p + 20,
                });
            }
        }
    }

    for &typ in &["EMA", "SMA"] {
        for &p in &[8usize, 20, 50, 100] {
            for &trans in TRANSFORMATIONS {
                for &mode in &["calc", "src"] {
                    for &conf in CONFLUENCES {
                        let trans_suffix = if trans.is_empty() { String::new() } else { format!("_{}_{}", trans, mode) };
                        let conf_suffix = if conf.is_empty() { String::new() } else { format!("_{}", conf) };
                        let name = format!("RSI_x_{}{}{}{}", typ, p, trans_suffix, conf_suffix);
                        specs.push(StrategySpec {
                            name, primary: Primary::RSI,
                            partner_kind: if typ == "EMA" { PartnerKind::EMA } else { PartnerKind::SMA },
                            partner_val: Some(p), macd_params: None,
                            transformation: transformation_from_str(trans),
                            transform_mode: if mode == "src" { TransformMode::Src } else { TransformMode::Calc },
                            confluence: confluence_from_str(conf),
                            stop_loss: 0.0, default_lb: p + 20,
                        });
                    }
                }
            }
        }
    }

    for &typ in &["SMA", "EMA"] {
        for &lvl in &[40usize, 50, 60] {
            let name = format!("RSI_{}_{}", typ, lvl);
            specs.push(StrategySpec {
                name, primary: Primary::RsiLevel,
                partner_kind: if typ == "EMA" { PartnerKind::EMA } else { PartnerKind::SMA },
                partner_val: Some(lvl), macd_params: None,
                transformation: Transformation::None,
                transform_mode: TransformMode::Calc,
                confluence: Confluence::None,
                stop_loss: 0.0, default_lb: 50,
            });
        }
    }

    for &(f, s) in &[(24usize, 52usize), (16, 42), (26, 68), (42, 110)] {
        for &trans in TRANSFORMATIONS {
            for &conf in CONFLUENCES {
                let trans_suffix = if trans.is_empty() { String::new() } else { format!("_{}", trans) };
                let conf_suffix = if conf.is_empty() { String::new() } else { format!("_{}", conf) };
                let name = format!("MACD({},{}){}{}", f, s, trans_suffix, conf_suffix);
                specs.push(StrategySpec {
                    name, primary: Primary::MACD, partner_kind: PartnerKind::EMA,
                    partner_val: None, macd_params: Some((f, s)),
                    transformation: transformation_from_str(trans),
                    transform_mode: TransformMode::Calc,
                    confluence: confluence_from_str(conf),
                    stop_loss: 0.0, default_lb: f,
                });
            }
        }
    }

    for &kind in &["SMA", "EMA"] {
        for &p in &[21usize, 55, 89] {
            for &trans in TRANSFORMATIONS {
                for &mode in &["calc", "src"] {
                    for &conf in CONFLUENCES {
                        let trans_suffix = if trans.is_empty() { String::new() } else { format!("_{}_{}", trans, mode) };
                        let conf_suffix = if conf.is_empty() { String::new() } else { format!("_{}", conf) };
                        let name = format!("STOCHK_{}_{}{}{}", kind, p, trans_suffix, conf_suffix);
                        specs.push(StrategySpec {
                            name, primary: Primary::StochK,
                            partner_kind: if kind == "EMA" { PartnerKind::EMA } else { PartnerKind::SMA },
                            partner_val: Some(p), macd_params: None,
                            transformation: transformation_from_str(trans),
                            transform_mode: if mode == "src" { TransformMode::Src } else { TransformMode::Calc },
                            confluence: confluence_from_str(conf),
                            stop_loss: 0.0, default_lb: p,
                        });
                    }
                }
            }
        }
    }

    for &typ in &["EMA", "SMA"] {
        for &p in &[50usize, 100, 200] {
            for &trans in TRANSFORMATIONS {
                for &mode in &["calc", "src"] {
                    for &conf in CONFLUENCES {
                        let trans_suffix = if trans.is_empty() { String::new() } else { format!("_{}_{}", trans, mode) };
                        let conf_suffix = if conf.is_empty() { String::new() } else { format!("_{}", conf) };
                        let name = format!("ATR_x_{}{}{}{}", typ, p, trans_suffix, conf_suffix);
                        specs.push(StrategySpec {
                            name, primary: Primary::ATR,
                            partner_kind: if typ == "EMA" { PartnerKind::EMA } else { PartnerKind::SMA },
                            partner_val: Some(p), macd_params: None,
                            transformation: transformation_from_str(trans),
                            transform_mode: if mode == "src" { TransformMode::Src } else { TransformMode::Calc },
                            confluence: confluence_from_str(conf),
                            stop_loss: 0.0, default_lb: 50,
                        });
                    }
                }
            }
        }
    }

    for &typ in &["EMA", "SMA"] {
        for &p in &[8usize, 20, 50, 100] {
            for &trans in TRANSFORMATIONS {
                for &mode in &["calc", "src"] {
                    for &conf in CONFLUENCES {
                        let trans_suffix = if trans.is_empty() { String::new() } else { format!("_{}_{}", trans, mode) };
                        let conf_suffix = if conf.is_empty() { String::new() } else { format!("_{}", conf) };
                        let name = format!("PPO_x_{}{}{}{}", typ, p, trans_suffix, conf_suffix);
                        specs.push(StrategySpec {
                            name, primary: Primary::PPO,
                            partner_kind: if typ == "EMA" { PartnerKind::EMA } else { PartnerKind::SMA },
                            partner_val: Some(p), macd_params: None,
                            transformation: transformation_from_str(trans),
                            transform_mode: if mode == "src" { TransformMode::Src } else { TransformMode::Calc },
                            confluence: confluence_from_str(conf),
                            stop_loss: 0.0, default_lb: p + 20,
                        });
                    }
                }
            }
        }
    }

    let base = specs;
    let mut final_specs: Vec<StrategySpec> = Vec::new();
    for spec in &base {
        for &sl in &stop_loss_values() {
            let mut s = spec.clone();
            s.stop_loss = sl;
            s.name = format!("{}_SL{}", s.name, sl as usize);
            final_specs.push(s);
        }
    }

    final_specs
}

fn get_completed(output_dir: &str) -> HashSet<String> {
    let mut done = HashSet::new();
    let base = Path::new(output_dir);
    if !base.is_dir() { return done; }
    if let Ok(groups) = fs::read_dir(base) {
        for group in groups.flatten() {
            if !group.path().is_dir() { continue; }
            if let Ok(strats) = fs::read_dir(group.path()) {
                for strat in strats.flatten() {
                    if !strat.path().is_dir() { continue; }
                    let strat_name = strat.file_name().to_string_lossy().to_string();
                    let done_file = strat.path().join("_done");
                    if done_file.exists() {
                        done.insert(strat_name);
                    }
                }
            }
        }
    }
    done
}


// ---------------------------------------------------------------------------
// Section 15: Main — Parallel Execution Model
//
// The entry point loads configuration (from config.toml if present, then CLI
// overrides), reads market data once into an Arc, generates the full strategy
// grid, skips already-completed strategies (by checking for `_done` marker
// files), and fans out the remaining work via rayon's parallel iterator.
// Each strategy runs independently, writing its own log and trade file.
// A progress counter is printed every 100 completions.
// ---------------------------------------------------------------------------

fn main() {
    let total_start = Instant::now();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--generate-config") {
        let sample = AppConfig::generate_sample_toml();
        if let Ok(mut f) = File::create("config.toml") {
            let _ = f.write_all(sample.as_bytes());
        }
        println!("Generated config.toml with default values.");
        return;
    }

    let cfg_file = if Path::new("config.toml").exists() { "config.toml" } else { "" };
    let app_cfg = if !cfg_file.is_empty() {
        println!("Loading configuration from config.toml");
        AppConfig::load_from_toml(cfg_file)
    } else {
        AppConfig::default()
    };
    APP_CONFIG.set(app_cfg).ok();
    let a = app();

    let use_bin = args.iter().any(|arg| arg == "--bin");
    let args_filtered: Vec<&String> = args.iter().filter(|arg| !arg.starts_with("--")).collect();
    let output_dir = if args_filtered.len() > 1 { args_filtered[1].as_str() } else { "strategy_outputs" };
    let csv_file = if args_filtered.len() > 2 { args_filtered[2].as_str() } else { &a.csv_file };
    let bc = if args_filtered.len() > 3 { args_filtered[3].parse::<usize>().expect("bad backtest_candles") } else { a.backtest_candles };
    let oc = if args_filtered.len() > 4 { args_filtered[4].parse::<usize>().expect("bad oos_candles") } else { a.oos_candles };
    let wt = if args_filtered.len() > 5 { args_filtered[5].parse::<usize>().expect("bad wfo_trigger_val") } else { a.wfo_trigger_val };
    let sl = if args_filtered.len() > 6 {
        args_filtered[6].split(',').map(|s| s.trim().parse::<f64>().expect("bad sl_value")).collect()
    } else { a.stop_loss_values.clone() };

    RUN_PARAMS.set(RunParams { backtest_candles: bc, oos_candles: oc, wfo_trigger_val: wt, sl_values: sl }).ok();

    let csv_path = csv_file.to_string();

    println!("Loading data from: {}", csv_path);
    println!("Config: BACKTEST_CANDLES={}, OOS_CANDLES={}, WFO_TRIGGER_VAL={}", bc, oc, wt);
    let load_start = Instant::now();
    let bars = load_ohlc(&csv_path);
    let bars = age_dataset(bars, AGE_DATASET);
    println!("Loaded {} bars in {:.2}s", bars.len(), load_start.elapsed().as_secs_f64());

    let bars = Arc::new(bars);

    let strategies = generate_all_strategies();
    println!("Generated {} strategy specs", strategies.len());

    let done = get_completed(output_dir);
    let pending: Vec<&StrategySpec> = strategies.iter()
        .filter(|s| !done.contains(&s.name))
        .collect();

    if !done.is_empty() {
        println!("[runner] Skipping {} already completed strategies", done.len());
    }
    println!("[runner] Running {} pending strategies on {} threads", pending.len(), rayon::current_num_threads());

    let completed = AtomicUsize::new(0);
    let total = pending.len();

    if use_bin { println!("[runner] Binary trade output mode enabled"); }

    pending.par_iter().for_each(|spec| {
        let _ = run_strategy(&bars, spec, output_dir, use_bin);
        let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
        if c % 100 == 0 || c == total {
            println!("[runner] Completed {}/{} strategies ({:.1}%)", c, total, c as f64 / total as f64 * 100.0);
        }
    });

    println!("\n[runner] All {} strategies completed in {:.1}s", total, total_start.elapsed().as_secs_f64());
    println!("[runner] Output root: {}", output_dir);
    if use_bin { println!("[runner] Use `convert_trades` to expand .bin files to CSV"); }
}
