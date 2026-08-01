# Regime-Aware Portfolio Optimizer

Detects **market regimes** with a Gaussian **Hidden Markov Model (HMM)** on asset
returns, then allocates capital by regime (risk-on vs. risk-off) and benchmarks
the result against a static **60/40** portfolio.

## What it does
- Fits a 2-state Gaussian HMM to return data and labels the high-volatility
  state as "risk-off"
- Shifts allocation defensively in turbulent regimes (80/20 → 20/80)
- Reports annualized return, volatility, and **Sharpe ratio** vs. a static
  benchmark

## Run it
```bash
pip install -r requirements.txt
python regime_optimizer.py
```
Bring real data with a CSV of daily returns:
```bash
python regime_optimizer.py --csv returns.csv   # columns: equity, bond
```

## Notes
Uses synthetic two-regime return data by default so the HMM detection and
allocation logic are fully reproducible offline. The framework — regime
detection → conditional allocation → risk-adjusted benchmarking — is the
transferable part.

_Author: Adrian Erlikhman_
