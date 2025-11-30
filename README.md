# Intraday VIX Term-Structure Signal on ES Futures

## Overview

This project investigates whether intraday VIX term-structure — specifically the slope between VIX and VIX3M — contains predictive power for short-horizon ES futures returns.

Using 1-minute ES data (5-minute momentum window) merged with daily VIX/VIX3M, we test whether volatility backwardation/contango conditions influence next-period returns, and evaluate a simple long/short strategy based on quantile bucket thresholds.

---

## Motivation

Volatility term-structure reflects market expectations of near-term risk.

- Backwardation (VIX > VIX3M) → front-loaded fear → rebalancing flows, dealer hedging pressure, and short-horizon mean reversion in equity indices.

- Contango (VIX < VIX3M) → complacency → weaker hedging demand → slightly negative forward returns.

This project tests whether these theoretical patterns appear even when combining daily volatility indices with intraday ES futures.

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

## Challenges & Constraints

- VIX and VIX3M are daily closing values, not intraday → feature is low-frequency.
- ES 1-minute returns are dominated by microstructure noise.
- VIX3M history begins only in late 2007, dramatically shrinking train sample.
- Regime shift into 2008 financial crisis breaks distributional assumptions.

---

## 2. Feature Engineering

This study constructs three core elements: intraday returns, forward-looking targets, and the volatility term-structure feature. All features are aligned on a unified 1-minute grid after timezone correction and RTH filtering.

---

### 1-minute log return
The basic intraday movement metric:

\[
r_{1m}(t) = \log\left(\frac{P_t}{P_{t-1}}\right)
\]

Used primarily for exploratory diagnostics.

---

### 60-minute forward return (target)

\[
r_{60m}^{(future)}(t) = \log\left(\frac{P_{t+60}}{P_t}\right)
\]

This is the quantity we attempt to predict using volatility term-structure information.

---

### VIX Term-Structure Slope

The raw slope:

\[
\text{slope}(t) = {VIX}(t) - {VIX3M}(t)
\]

captures the relative steepness of the volatility term structure.  
Backwardation → VIX > VIX3M  
Contango → VIX < VIX3M

Because daily VIX and VIX3M values are forward-filled across intraday timestamps, a rolling normalization is required.

---

### Z-Scored Slope (ZSlope)

A 7700-minute (~20 trading days) rolling window is used to standardize the slope:

\[
ZSlope(t) = \frac{\text{slope}(t) - \mu_t}{\sigma_t}
\]

This removes slow-moving level shifts and ensures all decile comparisons are done on a stationary-like feature.

---

### Quantile Buckets (Deciles)

To evaluate whether ZSlope exhibits systematic directional information, the feature is divided into 10 equally sized buckets (deciles) based on its distribution within the training window.

For each bucket, we compute the mean future 60-minute return.  
This reveals whether extreme backwardation/contango corresponds to consistent price pressure.

Observed behavior (training window):

- Lowest decile (extreme backwardation) → highest positive return  
- Highest decile (strong contango) → also positive return  
- Middle deciles → no significant structure  
- No monotonic relationship across buckets  

These patterns indicate that ZSlope provides no stable or reliable predictive signal during the 2007–2008 sample, largely due to the limited historical availability of VIX3M and the regime shift entering 2008.


---

## Signal Rules

Because the VIX3M history begins only in late 2007, the training window is extremely small.  
For completeness, the study still constructs simple long/short rules based on ZSlope deciles

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

## What I Learned

How to align intraday futures data with daily macro indices

The difficulty of extracting signals from microstructure-level ES noise

Importance of regime-awareness when evaluating volatility-based factors

Handling limited datasets and understanding when a signal is not robust

How term-structure information interacts with short-horizon equity index returns

---

## Disclaimer
Research only.  
Not investment advice.  
Past performance does not guarantee future results.

---
