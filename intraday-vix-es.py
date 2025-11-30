
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import time  


# -------------------------
# 1. Load ES intraday (1m, 24h) and restrict to RTH
# -------------------------

def load_es_intraday(path):
    """
    Load ES 1-minute data, set timestamp index, localize to US/Eastern,
    and keep only OHLCV columns.
    """
    df = pd.read_csv(path)
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Use timestamp as index
    df = df.set_index('timestamp').sort_index()
    
    # If timestamps have no timezone, assume they're in US/Eastern
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    
    # Keep only OHLCV, cast to float
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    print("Raw ES datetime range:", df.index[0], "→", df.index[-1])
    print("ES columns:", df.columns.tolist())
    return df


def restrict_rth(df, start="09:30", end="16:00"):
    """
    Keep only regular trading hours (RTH) for ES,
    09:30–16:00 US/Eastern.
    """
    df_rth = df.between_time(start, end).copy()
    print("ES RTH datetime range:", df_rth.index[0], "→", df_rth.index[-1])
    print("Number of ES RTH bars:", len(df_rth))
    return df_rth


# -------------------------
# 2. Load VIX & VIX3M daily
# -------------------------

def load_vix(path):
    """
    Load VIX daily, use CLOSE as VIX level.
    """
    df = pd.read_csv(path)
    # Your VIX file has format like '01/02/1990'
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%Y")
    df = df.set_index('timestamp').sort_index()
    
    vix = df[['CLOSE']].rename(columns={'CLOSE': 'VIX'})
    
    # Localize to US/Eastern at midnight
    if vix.index.tz is None:
        vix.index = vix.index.tz_localize('America/New_York')
    
    print("VIX daily date range:", vix.index[0].date(), "→", vix.index[-1].date())
    return vix


def load_vix3m(path):
    """
    Load VIX3M daily, use 'Price' column as VIX3M level.
    """
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    
    vix3m = df[['Price']].rename(columns={'Price': 'VIX3M'})
    
    if vix3m.index.tz is None:
        vix3m.index = vix3m.index.tz_localize('America/New_York')
    
    print("VIX3M daily date range:", vix3m.index[0].date(), "→", vix3m.index[-1].date())
    return vix3m


# -------------------------
# 3. Merge ES + VIX + VIX3M on a 1-minute RTH grid
# -------------------------

def merge_es_vix(v_es_rth, vix, vix3m):
    """
    Merge ES RTH 1m with daily VIX/VIX3M by date.
    Result: 1-min DataFrame with columns:
    ['open','high','low','close','volume','date','VIX','VIX3M']
    """
    df = v_es_rth.copy()
    # Add a date column (naive date, no timezone) for daily join
    df['date'] = df.index.date
    
    # Prepare daily VIX/VIX3M with 'date' column
    vix_daily = vix.copy()
    vix_daily['date'] = vix_daily.index.date
    
    vix3m_daily = vix3m.copy()
    vix3m_daily['date'] = vix3m_daily.index.date
    
    # Merge by date (no look-ahead within a day: each day gets that day's VIX)
    tmp = (
        df.reset_index()
          .merge(vix_daily[['date', 'VIX']], on='date', how='left')
          .merge(vix3m_daily[['date', 'VIX3M']], on='date', how='left')
          .set_index('timestamp')
          .sort_index()
    )
    
    print("\nMerged DataFrame columns:", tmp.columns.tolist())
    print("Any NaNs per column after merge?")
    print(tmp.isna().sum())
    return tmp


# -------------------------
# 4. Train / Test split by days
# -------------------------

def add_train_test_flag(df, train_frac=0.7):
    """
    Split by days: first 70% days = train, last 30% = test.
    Adds a column 'set' with values 'train' or 'test'.
    """
    df = df.copy()
    unique_days = np.array(sorted(df['date'].unique()))
    n_days = len(unique_days)
    n_train = int(n_days * train_frac)
    
    train_days = set(unique_days[:n_train])
    test_days  = set(unique_days[n_train:])
    
    df['set'] = np.where(df['date'].isin(train_days), 'train', 'test')
    
    print("\nNumber of trading days:", n_days)
    print("Train days:", len(train_days), "Test days:", len(test_days))
    print(df['set'].value_counts())
    return df


# -------------------------
# 5. Quick sanity plots (ES top, VIX bottom)
# -------------------------

def plot_phase1(df):
    """
    Top: ES close
    Bottom: VIX
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    ax1.plot(df.index, df['close'])
    ax1.set_title("ES 1-min Close (RTH)")
    ax1.set_ylabel("Price")
    
    ax2.plot(df.index, df['VIX'])
    ax2.set_title("VIX Daily Level (forward-filled within day)")
    ax2.set_ylabel("VIX")
    
    plt.tight_layout()
    plt.show()


# -------------------------
# 6. Run Phase 1
# -------------------------

# Adjust paths to where you saved the files
es_path = "/Users/lucy/intraday-vix-es/data/es_1min.csv"
vix_path  = "/Users/lucy/intraday-vix-es/data/VIX.csv"
vix3m_path = "/Users/lucy/intraday-vix-es/data/VIX3M.csv"

# 1) Load raw ES (24h) and restrict to RTH
es_raw = load_es_intraday(es_path)
es_rth = restrict_rth(es_raw, start="09:30", end="16:00")

# 2) Load VIX & VIX3M
vix   = load_vix(vix_path)
vix3m = load_vix3m(vix3m_path)

# 3) Merge on a 1-minute RTH grid
es_merged = merge_es_vix(es_rth, vix, vix3m)

# 4) Add train/test flag
es_merged = add_train_test_flag(es_merged, train_frac=0.7)

# 5) Sanity plots
plot_phase1(es_merged)


# ========================================
# PHASE 2 – FEATURE CONSTRUCTION
# ========================================

# -----------------------------
# 0. Parameters
# -----------------------------
K = 60        # horizon for future return label (in minutes)
W_days = 20   # rolling window length in "trading days" for ZSlope

# Estimate how many 1-min bars per trading day (should be ~390)
minutes_per_day = int(es_merged.groupby('date').size().median())
W = W_days * minutes_per_day   # rolling window length in minutes

print(f"Estimated minutes per day: {minutes_per_day}")
print(f"Rolling window W = {W} minutes (~{W_days} trading days)")

# Work on a copy to be explicit (optional)
df = es_merged.copy()

# -----------------------------
# 1. 1-minute log return r1_t
#    r1_t = log(close_t / close_{t-1})
#    (computed within each day only)
# -----------------------------
def intraday_log_return(series):
    """Log return within each day, leaving NaN at first bar of the day."""
    return np.log(series / series.shift(1))

# Recalculate 1-minute log return (correct alignment)
df['r1_1m'] = (
    df.groupby('date')['close']
      .transform(lambda x: np.log(x / x.shift(1)))
)

# Recalculate K-minute future return label (correct alignment)
df[f'rK_{K}_m'] = (
    df.groupby('date')['close']
      .transform(lambda x: np.log(x.shift(-K) / x))
)

# Recompute NaN diagnostics
print("\nNaNs after FIX:")
print(df[['r1_1m', f'rK_{K}_m']].isna().sum())

# Reassign back to main name
es_merged = df

# -----------------------------
# 2. Future K-minute log return rK_t (label)
#    rK_t = log(close_{t+K} / close_t),
#    computed within each day so it never crosses days.
# -----------------------------
def future_log_return(series, K):
    """Future K-minute log return within each day."""
    return np.log(series.shift(-K) / series)


# Last K rows of each day now have NaN in the label (no look-ahead across days).

# -----------------------------
# 3. VIX term-structure slope:
#    Slope_t = VIX_t - VIX3M_t
# -----------------------------
df['Slope'] = df['VIX'] - df['VIX3M']

# -----------------------------
# 4. Rolling z-score of Slope (ZSlope_t)
#    Using a rolling window of W minutes, past data only (no lookahead).
# -----------------------------
rolling_mean = df['Slope'].rolling(window=W, min_periods=W//4).mean()
rolling_std  = df['Slope'].rolling(window=W, min_periods=W//4).std()

df['ZSlope'] = (df['Slope'] - rolling_mean) / rolling_std

# -----------------------------
# 5. Quick sanity checks
# -----------------------------
print("\nHead with new features:")
print(df[['close', 'r1_1m', f'rK_{K}_m', 'Slope', 'ZSlope', 'set']].head(15))

# Replace original DataFrame if you want to keep using the same name:
es_merged = df

# ========================================
# PHASE 3 – SIGNAL RESEARCH (TRAIN ONLY)
# ========================================

# Work off a copy for safety
df = es_merged.copy()

#-------------------------------------------------
# 1. Use TRAIN set only + drop NaNs in features/labels
#-------------------------------------------------
train_df = df[df['set'] == 'train'].copy()

train_df = train_df.dropna(subset=['ZSlope', f'rK_{K}_m'])

print("Train rows after NaN removal:", len(train_df))

#-------------------------------------------------
# 2. Plot distributions
#-------------------------------------------------

# Histogram: raw slope
plt.figure(figsize=(10,4))
plt.hist(train_df['Slope'], bins=50)
plt.title("Histogram of Slope (VIX - VIX3M) — TRAIN")
plt.xlabel("Slope")
plt.ylabel("Count")
plt.show()

# Histogram: ZSlope
plt.figure(figsize=(10,4))
plt.hist(train_df['ZSlope'], bins=50)
plt.title("Histogram of ZSlope — TRAIN")
plt.xlabel("ZSlope")
plt.ylabel("Count")
plt.show()

#-------------------------------------------------
# 3. Pearson correlation
#-------------------------------------------------
corr = train_df['ZSlope'].corr(train_df[f'rK_{K}_m'])

print(f"Pearson correlation (ZSlope vs rK_{K}_m): {corr:.5f}")

#-------------------------------------------------
# 4. Scatter plot
#-------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(train_df['ZSlope'], train_df[f'rK_{K}_m'], alpha=0.3)
plt.title("ZSlope vs Future 60m Return (TRAIN)")
plt.xlabel("ZSlope")
plt.ylabel(f"rK_{K}_m")
plt.show()

#-------------------------------------------------
# 5. Bucket (quantile) analysis
#-------------------------------------------------

# Choose number of buckets
n_buckets = 10

# Create quantile buckets
train_df['z_bucket'] = pd.qcut(train_df['ZSlope'], q=n_buckets, labels=False)

# Group by bucket and compute stats
bucket_stats = train_df.groupby('z_bucket')[f'rK_{K}_m'].agg(
    mean='mean',
    count='count',
    std='std'
)

# Compute t-stat = mean / (std / sqrt(N))
bucket_stats['t_stat'] = (
    bucket_stats['mean'] /
    (bucket_stats['std'] / np.sqrt(bucket_stats['count']))
)

print("\nBucket statistics:")
print(bucket_stats)

# Plot mean future return by bucket
plt.figure(figsize=(10,4))
plt.plot(bucket_stats.index, bucket_stats['mean'], marker='o')
plt.title("Mean Future 60m Return by ZSlope Bucket (TRAIN)")
plt.xlabel("ZSlope Quantile Bucket (low → high)")
plt.ylabel("Mean rK")
plt.show()

#-------------------------------------------------
# 6. Intraday robustness (by hour)
#-------------------------------------------------

train_df['hour'] = train_df.index.hour

intraday_stats = (
    train_df
    .groupby('hour')[f'rK_{K}_m']
    .mean()
)

plt.figure(figsize=(10,4))
plt.plot(intraday_stats.index, intraday_stats.values, marker='o')
plt.title("Mean Future 60m Return by Hour of Day (TRAIN)")
plt.xlabel("Hour")
plt.ylabel("Mean rK")
plt.show()

#-------------------------------------------------
# 7. Time robustness (by month or year)
#-------------------------------------------------

train_df['year'] = train_df.index.year
train_df['month'] = train_df.index.month

time_stats = (
    train_df
    .groupby(['year', 'month'])[f'rK_{K}_m']
    .mean()
)


print("\nMean future returns by year-month:")
print(time_stats)


# ========================================
# PHASE 4 – RULE DISCOVERY (FROM TRAIN DATA)
# ========================================

import numpy as np
import pandas as pd

df = es_merged.copy()

# Use only train rows with valid feature + label
train_df = df[df['set'] == 'train'].copy()
train_df = train_df.dropna(subset=['ZSlope', f'rK_{K}_m'])

print("Train rows used for rule discovery:", len(train_df))

# -------------------------------------------------
# 1. Search over quantile thresholds for ZSlope
#    to separate "low" vs "high" regimes
# -------------------------------------------------

candidate_lows  = [0.10, 0.20, 0.30]   # bottom 10–30%
candidate_highs = [0.70, 0.80, 0.90]   # top   10–30%

best = None

for q_low in candidate_lows:
    for q_high in candidate_highs:
        if q_low >= q_high:
            continue
        
        low_thr  = train_df['ZSlope'].quantile(q_low)
        high_thr = train_df['ZSlope'].quantile(q_high)
        
        long_side  = train_df[train_df['ZSlope'] <= low_thr][f'rK_{K}_m']
        short_side = train_df[train_df['ZSlope'] >= high_thr][f'rK_{K}_m']
        
        n_long  = len(long_side)
        n_short = len(short_side)
        
        # Skip combinations with too few samples
        if n_long < 30 or n_short < 30:
            continue
        
        long_mean  = long_side.mean()
        short_mean = short_side.mean()
        edge = abs(long_mean - short_mean)
        
        candidate = {
            'q_low': q_low,
            'q_high': q_high,
            'low_thr': low_thr,
            'high_thr': high_thr,
            'long_mean': long_mean,
            'short_mean': short_mean,
            'n_long': n_long,
            'n_short': n_short,
            'edge': edge
        }
        
        if best is None or edge > best['edge']:
            best = candidate

print("\nBest quantile pair (based on |mean_long - mean_short|):")
print(best)

# -------------------------------------------------
# 2. Decide rule direction based on data
#    We LONG the side with higher mean rK,
#    and SHORT (if we want) the side with lower mean.
# -------------------------------------------------

if best['short_mean'] > best['long_mean']:
    # High ZSlope has higher future return -> bullish
    long_side  = 'high'
    short_side = 'low'
    long_thr_desc  = f"ZSlope >= {best['high_thr']:.4f} (q={best['q_high']:.2f})"
    short_thr_desc = f"ZSlope <= {best['low_thr']:.4f}  (q={best['q_low']:.2f})"
    long_thr_value  = float(best['high_thr'])
    short_thr_value = float(best['low_thr'])
else:
    # Low ZSlope has higher future return -> bullish
    long_side  = 'low'
    short_side = 'high'
    long_thr_desc  = f"ZSlope <= {best['low_thr']:.4f}  (q={best['q_low']:.2f})"
    short_thr_desc = f"ZSlope >= {best['high_thr']:.4f} (q={best['q_high']:.2f})"
    long_thr_value  = float(best['low_thr'])
    short_thr_value = float(best['high_thr'])

print("\nProposed ENTRY rules (TRAIN-based):")
print(f"- Go LONG  when {long_thr_desc}")
print(f"- Go SHORT when {short_thr_desc}")

# -------------------------------------------------
# 3. Neutral band for exits (unchanged)
# -------------------------------------------------
neutral_low_q  = 0.40
neutral_high_q = 0.60

neutral_low_thr  = train_df['ZSlope'].quantile(neutral_low_q)
neutral_high_thr = train_df['ZSlope'].quantile(neutral_high_q)

print("\nNeutral band for exits:")
print(f"- neutral_low  (q={neutral_low_q:.2f})  = {neutral_low_thr:.4f}")
print(f"- neutral_high (q={neutral_high_q:.2f}) = {neutral_high_thr:.4f}")

# -------------------------------------------------
# 4–5. Time / risk + RULES dict
# -------------------------------------------------
trade_start_time = "09:35"
trade_end_time   = "15:55"
max_holding_min  = K

RULES = {
    'K': K,
    'zslope': {
        'entry_long_side':   long_side,   # 'high' or 'low'
        'entry_short_side':  short_side,
        'entry_long_thresh': long_thr_value,
        'entry_short_thresh': short_thr_value,
        'entry_low_quantile':  best['q_low'],
        'entry_high_quantile': best['q_high'],
        'neutral_low_quantile':  neutral_low_q,
        'neutral_high_quantile': neutral_high_q,
        'neutral_low_threshold':  float(neutral_low_thr),
        'neutral_high_threshold': float(neutral_high_thr),
    },
    'time': {
        'start_time': trade_start_time,
        'end_time':   trade_end_time,
    },
    'risk': {
        'max_holding_minutes': max_holding_min,
        'one_position_only': True,
    }
}

print("\nFinal RULES dictionary (direction-consistent):")
for k, v in RULES.items():
    print(k, ":", v)
    
# ========================================
# PHASE 5 – SIMPLE BACKTEST ENGINE
# ========================================

import pandas as pd
from datetime import time

df_all = es_merged.copy().sort_index()

# Convenience: mark last bar of each day
df_all['is_last_bar'] = df_all['date'] != df_all['date'].shift(-1)


def run_backtest(df_in, RULES, label="test"):
    """
    Run a simple 1-unit notional backtest using ZSlope rules.
    
    Parameters
    ----------
    df_in : DataFrame
        Must contain at least ['close','ZSlope','date'].
    RULES : dict
        Output from Phase 4.
    label : str
        Name for this run (e.g. 'train', 'test').
    """
    df = df_in.copy().sort_index()

    # Extract rule parameters
    zcfg = RULES['zslope']
    tcfg = RULES['time']
    rcfg = RULES['risk']

    entry_long_thresh   = zcfg['entry_long_thresh']
    entry_short_thresh  = zcfg['entry_short_thresh']
    neutral_low         = zcfg['neutral_low_threshold']
    neutral_high        = zcfg['neutral_high_threshold']

    start_h, start_m = map(int, tcfg['start_time'].split(':'))
    end_h,   end_m   = map(int, tcfg['end_time'].split(':'))
    start_t = time(start_h, start_m)
    end_t   = time(end_h,   end_m)

    max_hold = rcfg['max_holding_minutes']

    # Backtest state
    position = 0          # -1 short, 0 flat, +1 long
    entry_price = None
    entry_time = None
    entry_idx = None      # integer index of bar where trade opened

    trade_log = []

    # We'll track equity at bar exits; start at 0
    equity = 0.0
    equity_curve = []

    # For easier indexing
    idx_list = list(df.index)
    closes = df['close'].values
    zs = df['ZSlope'].values
    dates = df['date'].values
    is_last_bar = df['is_last_bar'].values

    cost_per_side_bps = 1.0  # 1 bp per side

    for i, ts in enumerate(idx_list):
        price = closes[i]
        zval = zs[i]
        cur_date = dates[i]
        tod = ts.time()

        # Skip rows where ZSlope is NaN (no signal)
        if pd.isna(zval):
            # still need to force end-of-day exit
            neutral_now = False
        else:
            neutral_now = (neutral_low <= zval <= neutral_high)

        # --- EXIT LOGIC ---
        if position != 0:
            holding_bars = i - entry_idx
            exit_reason = None

            # 1) Max holding time
            if holding_bars >= max_hold:
                exit_reason = "max_hold"

            # 2) Neutral band
            if (exit_reason is None) and neutral_now:
                exit_reason = "neutral"

            # 3) End-of-day (force flat)
            if (exit_reason is None) and is_last_bar[i]:
                exit_reason = "eod"

            if exit_reason is not None:
                exit_price = price
                # PnL: position * (exit - entry)
                gross_pnl = position * (exit_price - entry_price)

                # Simple transaction costs: 1 bp per side on entry + exit
                notional = entry_price  # 1 unit * price
                cost = 2 * cost_per_side_bps / 10000.0 * notional
                pnl = gross_pnl - cost

                equity += pnl

                trade_log.append({
                    'label_run': label,
                    'entry_time': entry_time,
                    'exit_time': ts,
                    'direction': 'long' if position > 0 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pnl': gross_pnl,
                    'cost': cost,
                    'pnl': pnl,
                    'holding_minutes': holding_bars,
                    'exit_reason': exit_reason,
                    'date': cur_date
                })

                position = 0
                entry_price = None
                entry_time = None
                entry_idx = None

        # --- ENTRY LOGIC ---
        if position == 0:
            # Only enter within trading window
            if (start_t <= tod <= end_t) and (not pd.isna(zval)):
                # Long when ZSlope is in "long" regime (here: high ZSlope)
                if zval >= entry_long_thresh:
                    position = 1
                    entry_price = price
                    entry_time = ts
                    entry_idx = i

                # Short when ZSlope is in "short" regime (here: low ZSlope)
                elif zval <= entry_short_thresh:
                    position = -1
                    entry_price = price
                    entry_time = ts
                    entry_idx = i

        # Record equity curve at this timestamp
        equity_curve.append((ts, equity))

    equity_series = pd.Series(
        data=[e for _, e in equity_curve],
        index=[t for t, _ in equity_curve],
        name=f'equity_{label}'
    )

    trade_df = pd.DataFrame(trade_log)
    return trade_df, equity_series


# -------------------------------------------------
# Run backtest on TRAIN and TEST sets
# -------------------------------------------------

train_df_bt = df_all[df_all['set'] == 'train']
test_df_bt  = df_all[df_all['set'] == 'test']

trades_train, eq_train = run_backtest(train_df_bt, RULES, label="train")

trades_test, eq_test = run_backtest(test_df_bt, RULES, label="test")

print("\nNumber of trades (TRAIN):", len(trades_train))
print("Number of trades (TEST): ", len(trades_test))

print("\nSample of TEST trades:")
print(trades_test.head())

# Simple equity curve plot
plt.figure(figsize=(12,4))
eq_train.plot(label='Train')
eq_test.plot(label='Test')
plt.title("Equity Curve (Train vs Test)")
plt.xlabel("Time")
plt.ylabel("Cumulative P&L")
plt.legend()
plt.show()

# ========================================
# PHASE 6 – PERFORMANCE METRICS
# ========================================

import numpy as np

def performance_stats(trades: pd.DataFrame, equity: pd.Series, label=''):
    if trades.empty:
        print(f"\nNo trades for {label}")
        return {}

    stats = {}

    # Basic PnL vector
    pnl = trades['pnl']
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    stats['n_trades'] = len(trades)
    stats['win_rate'] = len(wins) / len(trades)
    stats['avg_win'] = wins.mean() if len(wins) > 0 else 0
    stats['avg_loss'] = losses.mean() if len(losses) > 0 else 0
    stats['expectancy'] = pnl.mean()
    stats['total_pnl'] = pnl.sum()

    # Profit factor
    stats['profit_factor'] = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan

    # Simple trade-based Sharpe
    if pnl.std() > 0:
        stats['sharpe_trade'] = pnl.mean() / pnl.std()
    else:
        stats['sharpe_trade'] = np.nan

    # Max Drawdown from equity curve
    eq = equity.values
    peaks = np.maximum.accumulate(eq)
    drawdown = eq - peaks
    stats['max_drawdown'] = drawdown.min()

    # Directional stats
    stats['n_longs']  = (trades['direction'] == 'long').sum()
    stats['n_shorts'] = (trades['direction'] == 'short').sum()

    stats['long_win_rate'] = (
        (trades[trades['direction'] == 'long']['pnl'] > 0).mean()
        if stats['n_longs'] > 0 else np.nan
    )
    stats['short_win_rate'] = (
        (trades[trades['direction'] == 'short']['pnl'] > 0).mean()
        if stats['n_shorts'] > 0 else np.nan
    )

    print(f"\n===== {label.upper()} PERFORMANCE =====")
    for k, v in stats.items():
        print(f"{k:20s}: {v}")

    return stats


# Run stats
stats_train = performance_stats(trades_train, eq_train, label='train')
stats_test  = performance_stats(trades_test,  eq_test,  label='test')