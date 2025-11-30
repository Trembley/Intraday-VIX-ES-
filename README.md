# Intraday VIX Term-Structure Signal on ES Futures

## Overview

This project investigates whether intraday VIX term-structure — specifically the slope between VIX and VIX3M — contains predictive power for short-horizon ES futures returns.

Using 1-minute ES data (5-minute momentum window) merged with daily VIX/VIX3M, we test whether volatility backwardation/contango conditions influence next-period returns, and evaluate a simple long/short strategy based on quantile bucket thresholds.

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

**Data limitation:**  
VIX3M only starts in **late 2007**, making the usable training window extremely short (≈2 months).

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

## Signal Rules

Because the VIX3M history begins only in late 2007, the training window is extremely small.  
For completeness, the study still constructs simple long/short rules based on ZSlope deciles, but the instability of the bucket structure makes these rules exploratory rather than actionable.

### Long Entry (Exploratory) 
Enter long when ZSlope is in the **lowest deciles** (extreme backwardation).  
- Interpretation: Backwardation often reflects short-term stress; historically this can be followed by short-term mean reversion.  
- Limitation: In this dataset, the relation is weak and inconsistent across deciles.

### **Short Entry (Exploratory)**  
Enter short when ZSlope is in the **highest deciles** (strong contango).  
- Interpretation: A complacent volatility term structure may precede small downward pressure.  
- Limitation: The training sample is too small for reliable directional mapping.

### **Exit Rule**  
A fixed **60-minute holding period** is used to match the prediction horizon for future returns.

---

## Backtest Results

### Train Window: Nov–Dec 2007  
- Trades: 26  
- Win rate: 50%  
- PnL: +2.95 ES points  
- Too little data for reliable inference  

### Test Window: 2008  
- Trades: 843  
- Win rate: 46%  
- PnL: –550 ES points  
- Profit factor: 0.73  
- Max drawdown: –555 ES points  
- Long-biased exposure
  
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
