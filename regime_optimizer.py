"""
Regime-Aware Portfolio Optimizer
================================
Detects market regimes with a Gaussian Hidden Markov Model (HMM) on asset
returns, then allocates capital by regime (risk-on vs. risk-off) — a simple
volatility-regime framework — and benchmarks it against a static 60/40 mix.

Runs offline on synthetic multi-regime return data. Use real returns with:
    python regime_optimizer.py --csv returns.csv   # columns: equity,bond (daily returns)
"""
import argparse
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def synth_returns(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):  # persistent regimes with rare switches
        regime[i] = regime[i - 1] if rng.random() > 0.02 else 1 - regime[i - 1]
    equity = np.where(regime == 0,
                      rng.normal(0.0006, 0.008, n),   # calm bull
                      rng.normal(-0.0004, 0.020, n))  # volatile bear
    bond = rng.normal(0.0002, 0.003, n)
    return pd.DataFrame({"equity": equity, "bond": bond})


def ann_stats(r):
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol else 0.0
    return ann_ret, ann_vol, sharpe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", help="CSV with 'equity' and 'bond' daily-return columns")
    args = p.parse_args()

    rets = pd.read_csv(args.csv) if args.csv else synth_returns()

    # --- fit HMM on equity returns and label the high-vol state "risk-off" ---
    X = rets[["equity"]].values
    hmm = GaussianHMM(n_components=2, covariance_type="full",
                      n_iter=200, random_state=0)
    hmm.fit(X)
    states = hmm.predict(X)
    vols = [X[states == s].std() for s in range(2)]
    risk_off = int(np.argmax(vols))
    print(f"Detected 2 regimes | risk-off state = {risk_off} "
          f"(daily vol {vols[risk_off]:.4f} vs {vols[1 - risk_off]:.4f})")

    # --- regime-conditional allocation vs. static benchmark ---
    w_eq = np.where(states == risk_off, 0.20, 0.80)   # de-risk in turbulent regime
    port = w_eq * rets["equity"].values + (1 - w_eq) * rets["bond"].values
    bench = 0.60 * rets["equity"].values + 0.40 * rets["bond"].values

    print("\n--- Annualized performance ---")
    for name, series in [("Regime-aware", port), ("Static 60/40", bench)]:
        a, v, s = ann_stats(pd.Series(series))
        print(f"{name:14s} return={a:7.2%}  vol={v:6.2%}  Sharpe={s:5.2f}")


if __name__ == "__main__":
    main()
