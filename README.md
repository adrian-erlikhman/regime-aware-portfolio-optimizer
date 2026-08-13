# Regime-Aware Portfolio Optimizer

Two-state Gaussian hidden Markov model on daily equity returns. Capital in the risky leg is sized conditional on the decoded volatility regime and benchmarked against a static 60/40 on identical return streams. Fully reproducible offline on a synthetic multi-regime generator; accepts a real return series via `--csv`.

![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white) ![hmmlearn](https://img.shields.io/badge/hmmlearn-GaussianHMM-EE4C2C) ![License](https://img.shields.io/badge/license-MIT-black)

```mermaid
flowchart LR
    R[daily returns r_t] --> F[Baum-Welch EM<br/>fit theta]
    F --> V[Viterbi decode<br/>state path s_t]
    V --> L[label risk-off =<br/>argmax_k Var r|s=k]
    L --> W[regime-conditional<br/>weights w_eq t]
    W --> P[portfolio return]
    P --> M[risk/return vs 60/40]
```

## Problem

A static 60/40 holds constant weights regardless of the latent state of the market. Equity volatility is regime-dependent and persistent (volatility clustering), so a fixed mix systematically over-allocates risk in turbulent regimes and drags risk-adjusted return. The task is to infer the unobserved regime from returns alone and size the risky leg accordingly.

## Model

Returns $r_t \in \mathbb{R}$ are emissions of a discrete latent state $s_t \in \{0,1\}$ with first-order Markov dynamics:

$$s_t \mid s_{t-1} \sim \mathrm{Categorical}(A_{s_{t-1},\cdot}), \qquad r_t \mid s_t=k \sim \mathcal{N}(\mu_k, \Sigma_k)$$

Parameters $\theta=\{\pi, A, \mu, \Sigma\}$ are estimated by Baum-Welch (EM), maximizing the incomplete-data log-likelihood $\log p(r_{1:T}\mid\theta)$ via the forward-backward recursions. The maximum-likelihood state path $\hat s_{1:T}=\arg\max_{s_{1:T}} p(s_{1:T}\mid r_{1:T},\theta)$ is recovered by Viterbi. The high-volatility state is identified as $\text{risk-off}=\arg\max_k \operatorname{std}(r_t\mid \hat s_t=k)$.

Configuration: `GaussianHMM(n_components=2, covariance_type="full", n_iter=200)`.

## Allocation

$$w_\text{eq}(t)=\begin{cases}0.20 & \hat s_t=\text{risk-off}\\ 0.80 & \text{otherwise}\end{cases}\qquad r_\text{port}(t)=w_\text{eq}(t)\,r_\text{eq}(t)+\bigl(1-w_\text{eq}(t)\bigr)\,r_\text{bond}(t)$$

Benchmark: $r_\text{bench}(t)=0.60\,r_\text{eq}(t)+0.40\,r_\text{bond}(t)$.

## Metrics

`metrics.py` (NumPy only), annualized at 252 trading days:

$$\mathrm{CAGR}=\Bigl(\textstyle\prod_t(1+r_t)\Bigr)^{252/n}-1,\quad \mathrm{Sharpe}=\frac{\bar r}{\sigma_r}\sqrt{252},\quad \mathrm{Sortino}=\frac{\bar r}{\sigma_r^{-}}\sqrt{252}$$

$$\mathrm{MaxDD}=\min_t\Bigl(\frac{E_t}{\max_{\tau\le t}E_\tau}-1\Bigr),\quad \mathrm{Calmar}=\frac{\mathrm{CAGR}}{|\mathrm{MaxDD}|},\qquad E_t=\textstyle\prod_{\tau\le t}(1+r_\tau)$$

where $\sigma_r^{-}$ is the downside deviation (std over $r_t<0$).

## Results

Synthetic generator, `seed=7`, deterministic. Decoded risk-off daily vol $0.0191$ vs $0.0079$ risk-on.

| Strategy       | CAGR    | Vol     | Sharpe | Sortino | MaxDD    | Calmar |
|----------------|---------|---------|--------|---------|----------|--------|
| Regime-aware   |  3.76%  |  8.15%  |  0.49  |  0.84   | −13.70%  |  0.27  |
| Static 60/40   | −2.07%  | 15.76%  | −0.05  | −0.08   | −37.48%  | −0.06  |

On identical return streams the regime overlay roughly halves volatility and max drawdown and moves Sharpe from negative to positive.

## Reproduce

```bash
pip install -r requirements.txt
python regime_optimizer.py                      # synthetic multi-regime demo
python regime_optimizer.py --csv returns.csv    # columns: equity,bond (daily returns)
```

## Limitations

- Regimes are decoded in-sample. A walk-forward / expanding-window decode is required before any out-of-sample or tradable interpretation; the current figures quantify the model's descriptive fit, not a live backtest.
- Two-state Gaussian emissions ignore heavy tails and jumps. Student-$t$ emissions or $\ge 3$ states are the natural extensions.
- Synthetic returns are illustrative; real-data figures will differ.

MIT · synthetic data for reproducibility.
