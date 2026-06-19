"""
Robustness Check 1 — Alternative Distress Quantiles
====================================================
Re-runs CoVaR at q = 0.01, 0.05, 0.10 and verifies that
|ΔCoVaR(USDe)| > |ΔCoVaR(USDC)| holds across all thresholds.

Input : merged_logreturns.csv
Output: covar_results/robustness_quantiles.csv

Run: python robustness_quantiles.py
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import os

os.makedirs("covar_results", exist_ok=True)

lr = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
lr = lr[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
lr.columns = ["BTC","ETH","USDe","USDC"]

PAIRS      = [("BTC","USDe"),("BTC","USDC"),("ETH","USDe"),("ETH","USDC")]
QUANTILES  = [0.01, 0.05, 0.10]   # distress quantiles to test
Q_MED      = 0.50
N_BOOT     = 1000
np.random.seed(42)

print(f"N = {len(lr)}\n")
rows = []

for q_dist in QUANTILES:
    print(f"{'='*58}")
    print(f"Distress quantile q = {q_dist}")
    print(f"{'='*58}")

    for crypto, stable in PAIRS:
        x = lr[crypto].values
        y = lr[stable].values
        X = np.column_stack([np.ones(len(x)), x])
        n = len(y)

        r_dist = QuantReg(y, X).fit(q=q_dist, vcov="iid")
        r_med  = QuantReg(y, X).fit(q=Q_MED,  vcov="iid")

        covar_dist = r_dist.params[0] + r_dist.params[1] * x
        covar_med  = r_med.params[0]  + r_med.params[1]  * x
        delta      = covar_dist - covar_med
        mean_delta = float(np.mean(delta))

        # Bootstrap CI
        boot = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx      = np.random.choice(n, n, replace=True)
            yb, Xb   = y[idx], X[idx]
            try:
                rb_d = QuantReg(yb, Xb).fit(q=q_dist, vcov="iid")
                rb_m = QuantReg(yb, Xb).fit(q=Q_MED,  vcov="iid")
                db   = (rb_d.params[0] + rb_d.params[1]*Xb[:,1]) - \
                       (rb_m.params[0] + rb_m.params[1]*Xb[:,1])
                boot[b] = np.mean(db)
            except Exception:
                boot[b] = np.nan

        boot  = boot[~np.isnan(boot)]
        ci_lo = np.percentile(boot, 2.5)
        ci_hi = np.percentile(boot, 97.5)
        sig   = ci_lo > 0 or ci_hi < 0
        stars = "***" if r_dist.pvalues[1]<0.01 else "**" if r_dist.pvalues[1]<0.05 \
                else "*" if r_dist.pvalues[1]<0.10 else ""

        print(f"  {crypto}→{stable}: ΔCoVaR={mean_delta:+.6f}  "
              f"CI[{ci_lo:+.6f},{ci_hi:+.6f}]  "
              f"β_dist={r_dist.params[1]:+.6f}{stars}  "
              f"{'sig ✓' if sig else 'not sig'}")

        rows.append({
            "q_distress": q_dist,
            "pair":       f"{crypto}→{stable}",
            "crypto":     crypto,
            "stable":     stable,
            "beta_dist":  round(r_dist.params[1], 6),
            "pval_dist":  round(r_dist.pvalues[1], 4),
            "mean_deltacovar": round(mean_delta, 6),
            "ci_lo":      round(ci_lo, 6),
            "ci_hi":      round(ci_hi, 6),
            "significant":sig,
        })

results = pd.DataFrame(rows)
results.to_csv("covar_results/robustness_quantiles.csv", index=False)

# Summary: does USDe > USDC ranking hold across all quantiles?
print(f"\n{'='*58}")
print("Ranking check: |ΔCoVaR(USDe)| > |ΔCoVaR(USDC)| across quantiles")
print(f"{'='*58}")
for q_dist in QUANTILES:
    sub = results[results["q_distress"] == q_dist]
    for crypto in ["BTC","ETH"]:
        usde = abs(sub[(sub.crypto==crypto)&(sub.stable=="USDe")]["mean_deltacovar"].values[0])
        usdc = abs(sub[(sub.crypto==crypto)&(sub.stable=="USDC")]["mean_deltacovar"].values[0])
        holds = usde > usdc
        print(f"  q={q_dist}  {crypto}: |ΔCoVaR(USDe)|={usde:.6f}  "
              f"|ΔCoVaR(USDC)|={usdc:.6f}  "
              f"{'✓ holds' if holds else '✗ fails'}")

print("\nSaved → covar_results/robustness_quantiles.csv")
