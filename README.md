# order-flow-signal
Order flow vs price dislocation signal (TSLA intraday)

# Order Flow vs Price Dislocation Signal

## Overview
This project studies the relationship between short-term order flow and price movement.
The goal is to detect situations where aggressive buying or selling pressure is not yet reflected in price.

## Research Question
Can signed order flow predict short-horizon future returns when price has not yet adjusted?

## Intuition
When there is strong aggressive buying but price barely moves up, this may indicate latent upward pressure.
Similarly, strong selling pressure without price decline may indicate latent downward pressure.

## Data
- Instrument: TSLA
- Frequency: Intraday (tick / 1s / 1min bars)
- Fields used:
  - Trade price
  - Trade size
  - Bid / Ask
  - Mid price

## Methodology

### Trade Classification
Trades are classified using the Lee–Ready algorithm:

- Buy-initiated if trade price > mid price
- Sell-initiated if trade price < mid price

### Signal Construction
We construct:

- Net Order Flow  
- Realized Price Movement  

Final signal:

Sₜ = Normalized(Flowₜ) − Normalized(ΔPₜ)

## Results
Key findings:
- Positive signal values tend to be followed by positive returns
- Negative signal values tend to be followed by negative returns
- Predictive power is modest but statistically consistent

## Repository Structure

