"""
CoVaR and Delta-CoVaR
=====================
Quantile regression of stablecoin returns on crypto returns.
Primary: BTC -> USDe and BTC -> USDC (comparison)
Secondary: ETH -> USDe and ETH -> USDC

Input : merged_logreturns.csv
Output: covar_results/covar_summary.csv
        covar_results/deltacovar_series.csv

Run: python covar.py 
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from scipy import stats
import os

os.makedirs("covar_results", exist_ok=True)

# Load
lr = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
lr = lr[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
lr.columns = ["BTC","ETH","USDe","USDC"]

PAIRS  = [("BTC","USDe"),("BTC","USDC"),("ETH","USDe"),("ETH","USDC")]
Q_DIST = 0.05   # distress quantile
Q_MED  = 0.50   # median (normal) quantile
N_BOOT = 1000
ALPHA  = 0.05
np.random.seed(42)

print(f"N = {len(lr)}  |  {lr.index[0].date()} → {lr.index[-1].date()}\n")

summary_rows = []
delta_df     = pd.DataFrame(index=lr.index)

print("=" * 60)
print("CoVaR via quantile regression")
print("=" * 60)

for crypto, stable in PAIRS:
    x = lr[crypto].values
    y = lr[stable].values
    X = np.column_stack([np.ones(len(x)), x])
    n = len(y)

    # Fit at distress and median quantile
    r05 = QuantReg(y, X).fit(q=Q_DIST, vcov="iid")
    r50 = QuantReg(y, X).fit(q=Q_MED,  vcov="iid")

    a05, b05 = r05.params
    a50, b50 = r50.params

    # CoVaR time series
    covar_05 = a05 + b05 * x
    covar_50 = a50 + b50 * x
    delta     = covar_05 - covar_50   # ΔCoVaR

    delta_df[f"DCoVaR_{crypto}_{stable}"] = delta

    mean_delta = float(np.mean(delta))

    # Bootstrap CI on mean ΔCoVaR
    boot = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx  = np.random.choice(n, n, replace=True)
        yb, Xb = y[idx], X[idx]
        try:
            rb05 = QuantReg(yb, Xb).fit(q=Q_DIST, vcov="iid")
            rb50 = QuantReg(yb, Xb).fit(q=Q_MED,  vcov="iid")
            db   = (rb05.params[0] + rb05.params[1]*Xb[:,1]) - \
                   (rb50.params[0] + rb50.params[1]*Xb[:,1])
            boot[b] = np.mean(db)
        except Exception:
            boot[b] = np.nan

    boot    = boot[~np.isnan(boot)]
    ci_lo   = np.percentile(boot, 2.5)
    ci_hi   = np.percentile(boot, 97.5)
    sig     = ci_lo > 0 or ci_hi < 0    # CI excludes zero

    stars = "***" if r05.pvalues[1]<0.01 else "**" if r05.pvalues[1]<0.05 \
            else "*" if r05.pvalues[1]<0.10 else ""

    print(f"\n  {crypto} → {stable}")
    print(f"    β(q=0.05) = {b05:+.6f}  p={r05.pvalues[1]:.4f}{stars}")
    print(f"    β(q=0.50) = {b50:+.6f}  p={r50.pvalues[1]:.4f}")
    print(f"    Mean ΔCoVaR = {mean_delta:+.6f}")
    print(f"    95% CI [{ci_lo:+.6f}, {ci_hi:+.6f}]  "
          f"{'significant ✓' if sig else 'not significant'}")

    summary_rows.append({
        "pair":         f"{crypto}→{stable}",
        "beta_05":      round(b05, 6),
        "pval_05":      round(r05.pvalues[1], 4),
        "beta_50":      round(b50, 6),
        "pval_50":      round(r50.pvalues[1], 4),
        "mean_deltacovar": round(mean_delta, 6),
        "ci_lo":        round(ci_lo, 6),
        "ci_hi":        round(ci_hi, 6),
        "significant":  sig,
    })

# Save
summary = pd.DataFrame(summary_rows)
summary.to_csv("covar_results/covar_summary.csv", index=False)
delta_df.to_csv("covar_results/deltacovar_series.csv")

# H2: USDe more exposed than USDC?
print("\n" + "=" * 60)
print("H2: |ΔCoVaR(USDe)| vs |ΔCoVaR(USDC)|")
print("=" * 60)
for crypto in ["BTC","ETH"]:
    d_usde = delta_df[f"DCoVaR_{crypto}_USDe"].values
    d_usdc = delta_df[f"DCoVaR_{crypto}_USDC"].values
    t, p   = stats.ttest_ind(np.abs(d_usde), np.abs(d_usdc))
    larger = np.mean(np.abs(d_usde)) > np.mean(np.abs(d_usdc))
    print(f"  {crypto}: |ΔCoVaR(USDe)|={np.mean(np.abs(d_usde)):.6f}  "
          f"|ΔCoVaR(USDC)|={np.mean(np.abs(d_usdc)):.6f}  "
          f"t={t:.3f}  p={p:.4f}  "
          f"{'USDe > USDC ✓' if larger else 'USDC >= USDe'}")

print("\nSaved → covar_results/covar_summary.csv")
print("Saved → covar_results/deltacovar_series.csv")
"""
RESULTS: 

N = 801  |  2024-02-21 → 2026-05-01

============================================================
CoVaR via quantile regression
============================================================

  BTC → USDe
    β(q=0.05) = +0.011306  p=0.0000***
    β(q=0.50) = +0.005517  p=0.0000
    Mean ΔCoVaR = -0.000685
    95% CI [-0.000807, -0.000573]  significant ✓

  BTC → USDC
    β(q=0.05) = -0.000382  p=0.4135
    β(q=0.50) = +0.000006  p=0.9803
    Mean ΔCoVaR = -0.000226
    95% CI [-0.000268, -0.000207]  significant ✓

  ETH → USDe
    β(q=0.05) = +0.005193  p=0.0090***
    β(q=0.50) = +0.003083  p=0.0000
    Mean ΔCoVaR = -0.000738
    95% CI [-0.000871, -0.000606]  significant ✓

  ETH → USDC
    β(q=0.05) = -0.000331  p=0.3195
    β(q=0.50) = -0.000019  p=0.9042
    Mean ΔCoVaR = -0.000229
    95% CI [-0.000269, -0.000208]  significant ✓

============================================================
H2: |ΔCoVaR(USDe)| vs |ΔCoVaR(USDC)|
============================================================
  BTC: |ΔCoVaR(USDe)|=0.000685  |ΔCoVaR(USDC)|=0.000226  t=87.103  p=0.0000  USDe > USDC ✓
  ETH: |ΔCoVaR(USDe)|=0.000738  |ΔCoVaR(USDC)|=0.000229  t=181.911  p=0.0000  USDe > USDC ✓

"""