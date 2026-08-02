"""
Performance metrics for backtested daily-return series.
Pure NumPy — no heavy dependencies. Used by regime_optimizer.py to score strategies.
"""
import numpy as np


def sharpe(r, freq=252, rf=0.0):
    r = np.asarray(r, float) - rf / freq
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 else float(r.mean() / sd * np.sqrt(freq))


def sortino(r, freq=252, rf=0.0):
    r = np.asarray(r, float) - rf / freq
    downside = r[r < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else 0.0
    return 0.0 if dd == 0 else float(r.mean() / dd * np.sqrt(freq))


def annual_return(r, freq=252):
    r = np.asarray(r, float)
    return float((1 + r).prod() ** (freq / len(r)) - 1)


def annual_vol(r, freq=252):
    return float(np.asarray(r, float).std(ddof=1) * np.sqrt(freq))


def max_drawdown(r):
    eq = (1 + np.asarray(r, float)).cumprod()
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def calmar(r, freq=252):
    mdd = abs(max_drawdown(r))
    return 0.0 if mdd == 0 else annual_return(r, freq) / mdd


def summary(r, freq=252):
    """Return a dict of the headline risk/return metrics for a return series."""
    return {
        "CAGR": annual_return(r, freq),
        "Vol": annual_vol(r, freq),
        "Sharpe": sharpe(r, freq),
        "Sortino": sortino(r, freq),
        "MaxDD": max_drawdown(r),
        "Calmar": calmar(r, freq),
    }
