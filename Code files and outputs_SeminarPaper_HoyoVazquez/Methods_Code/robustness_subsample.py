"""
Robustness Check 2 — Sub-sample Stability
==========================================
Splits sample at midpoint and re-runs CoVaR in each half.
Verifies BTC->USDe significant and USDe>USDC in both sub-periods.

Input : merged_logreturns.csv
Output: covar_results/robustness_subsample.csv

Run: python robustness_subsample.py
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import os

os.makedirs("covar_results", exist_ok=True)

lr = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
lr = lr[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
lr.columns = ["BTC","ETH","USDe","USDC"]

PAIRS  = [("BTC","USDe"),("BTC","USDC"),("ETH","USDe"),("ETH","USDC")]
Q_DIST = 0.05
Q_MED  = 0.50
N_BOOT = 1000
np.random.seed(42)

# Split at midpoint
mid   = len(lr) // 2
split = lr.index[mid]
subsamples = {
    f"Period 1 ({lr.index[0].date()} – {split.date()})": lr.iloc[:mid],
    f"Period 2 ({lr.index[mid].date()} – {lr.index[-1].date()})": lr.iloc[mid:],
}
print(f"Full sample: {len(lr)} obs")
print(f"Split date : {split.date()}  |  Period 1: {mid} obs  |  Period 2: {len(lr)-mid} obs\n")

rows = []

for period_label, sub in subsamples.items():
    print(f"{'='*58}")
    print(f"{period_label}")
    print(f"{'='*58}")

    for crypto, stable in PAIRS:
        x = sub[crypto].values
        y = sub[stable].values
        X = np.column_stack([np.ones(len(x)), x])
        n = len(y)

        r_dist = QuantReg(y, X).fit(q=Q_DIST, vcov="iid")
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
                rb_d = QuantReg(yb, Xb).fit(q=Q_DIST, vcov="iid")
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
              f"β={r_dist.params[1]:+.6f}{stars}  "
              f"{'sig ✓' if sig else 'not sig'}")

        rows.append({
            "period":     period_label,
            "pair":       f"{crypto}→{stable}",
            "crypto":     crypto,
            "stable":     stable,
            "n":          n,
            "beta_dist":  round(r_dist.params[1], 6),
            "pval_dist":  round(r_dist.pvalues[1], 4),
            "mean_deltacovar": round(mean_delta, 6),
            "ci_lo":      round(ci_lo, 6),
            "ci_hi":      round(ci_hi, 6),
            "significant":sig,
        })

results = pd.DataFrame(rows)
results.to_csv("covar_results/robustness_subsample.csv", index=False)

# Ranking check per period
print(f"\n{'='*58}")
print("Ranking check: |ΔCoVaR(USDe)| > |ΔCoVaR(USDC)| per period")
print(f"{'='*58}")
for period_label in subsamples:
    sub_res = results[results["period"] == period_label]
    for crypto in ["BTC","ETH"]:
        usde = abs(sub_res[(sub_res.crypto==crypto)&(sub_res.stable=="USDe")]["mean_deltacovar"].values[0])
        usdc = abs(sub_res[(sub_res.crypto==crypto)&(sub_res.stable=="USDC")]["mean_deltacovar"].values[0])
        sig_usde = sub_res[(sub_res.crypto==crypto)&(sub_res.stable=="USDe")]["significant"].values[0]
        holds = usde > usdc
        print(f"  {period_label[:8]}  {crypto}: "
              f"|ΔCoVaR(USDe)|={usde:.6f}{'*' if sig_usde else ''}  "
              f"|ΔCoVaR(USDC)|={usdc:.6f}  "
              f"{'✓ holds' if holds else '✗ fails'}")

print("\nSaved → covar_results/robustness_subsample.csv")
