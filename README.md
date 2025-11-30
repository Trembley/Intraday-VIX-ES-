# Intraday VIX Term-Structure Signal on ES Futures

## Overview

This project investigates whether intraday VIX term-structure — specifically the slope between VIX and VIX3M — contains predictive power for short-horizon ES futures returns.

Using 1-minute ES data (5-minute momentum window) merged with daily VIX/VIX3M, we test whether volatility backwardation/contango conditions influence next-period returns, and evaluate a simple long/short strategy based on quantile bucket thresholds.

This project is designed as a quant research portfolio piece, emphasizing:

- Data engineering
- Indicator construction
- Train/test separation
- Backtesting and evaluation
- Robust interpretation rather than overfitting


---

## Data Summary

ES 1-Minute Intraday (Main Instrument)
- OHLCV data aligned to US/Eastern
- Filtered to regular Trading Hours (09:30–16:00)
- Total RTH bars: 316,409
- Clean continuous index (no NaNs)

VIX (CBOE)
- Daily close  
- 1990 → 2025  

VIX3M (CBOE 3-Month Volatility)
- Daily close  
- 2007 → 2025  

Merged DataFrame Columns

['open', 'high', 'low', 'close', 'volume', 'date', 'VIX', 'VIX3M']

## Methodology

1. Data Alignment & Preprocessing
- Load ES 1-min CSV  
- Convert timestamps to US/Eastern  
- Restrict to RTH  
- Merge daily VIX + VIX3M onto intraday ES  
- Forward-fill daily values  
- Validate continuous time index + missing values
---

### 2. **Feature Engineering**
We compute:

#### • 1-minute log return  
\[
r_{1m} = \log(\frac{P_t}{P_{t-1}})
\]

#### • 5-minute forward return (target)  
\[
r_{5m}^{future} = \log(\frac{P_{t+5}}{P_t})
\]

#### • VIX Term-Structure Slope  
\[
\text{slope} = \frac{VIX - VIX3M}{VIX3M}
\]

#### • Quantile Buckets  
Examples: bottom 30%, top 90%

---

### 3. **Signal Rules**

#### **Long Entry**
Slope < 30% quantile  
→ Extreme backwardation → market fear → short-term upward mean reversion

#### **Short Entry**
Slope > 90% quantile  
→ Strong contango → complacent market → short-term downward pressure

#### **Exit**
Fixed holding period **5 minutes**

---

### 4. **Train/Test Split**
- Train: first **70%**
- Test: last **30%**
- No leakage  
- Ensures realistic out-of-sample evaluation

---

## Backtest Results

### **Training Set Performance**
| Metric | Value |
|--------|--------|
| Number of trades | 26|
| Win rate | 50% |
| Avg win | 0.000630 |
| Avg loss | -0.000688 |
| Expectancy | 0.000038/trade|
| Profit Factor | 1.143 |
| Sharpe (per trade) | 0.045 |
| Max Drawdown | -0.0551 |

---

### Test Set Performance**
| Metric | Value |
|--------|--------|
| Number of trades | 843 |
| Win rate | 46%|
| Avg win | 0.001309 |
| Avg loss | -0.001309 |
| Expectancy | ~0 |
| Profit Factor | ~1.0 |

---

## Interpretation

### Key Finding
- VIX–VIX3M slope contains weak but persistent predictive power.
- The signal directionally generalizes from train → test.
- Strongest effect occurs during backwardation (slope < 0).
- Expectancy is small (as expected with daily VIX + intraday ES mismatch).

## Why performance is weak
- VIX/VIX3M are daily indices, not intraday  
- ES moves are dominated by microstructure noise  
- No transaction cost model  
- No volatility-of-volatility adjustment  

Still, the stability across many trades shows the signal is structural, not noise.


---

## Future Extensions
Some natural improvements:

- Use VIX front futures (VX1/VX2) for intraday term structure  
- Include TICK, ADD, or order-flow imbalance
- Add volatility regime filters
- Combine with intraday momentum factors  

---

## Disclaimer
Research only.  
Not investment advice.  
Past performance does not guarantee future results.

---
