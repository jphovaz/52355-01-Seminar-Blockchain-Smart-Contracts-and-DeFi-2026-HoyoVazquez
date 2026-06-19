"""
Granger Causality Tests — VAR(2)
=================================
Tests whether past values of one variable significantly help predict
another, above and beyond that variable's own past values.

H0: X does NOT Granger-cause Y  (all coefficients on X lags in Y's equation = 0)
H1: X DOES Granger-cause Y     (at least one coefficient is significant)

Tests all directional pairs in the 4-variable system (12 pairs total).

Input : merged_logreturns.csv
Output: var_results/granger_results.csv

Run: python granger_causality.py  (Granger on Log Returns)
"""

import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import os

os.makedirs("var_results", exist_ok=True)

# ── Load and fit ───────────────────────────────────────────────────────────────
df = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
df = df[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
df.columns = ["BTC","ETH","USDe","USDC"]

P      = 2
fitted = VAR(df).fit(maxlags=P, ic=None, trend="c")

variables = list(df.columns)
ALPHA     = 0.05

# ── Run all directional Granger tests ─────────────────────────────────────────
"""
fitted.test_causality(caused, causing, kind="f") runs an F-test asking:
  "Do lags of [causing] jointly help predict [caused]?"
  caused  = the equation being predicted (Y)
  causing = the variable whose lags we are testing (X)
"""

print("=" * 62)
print(f"Granger Causality Tests — VAR({P})")
print("H0: [causing] does NOT Granger-cause [caused]")
print("=" * 62)

rows = []
for caused in variables:
    for causing in variables:
        if caused == causing:
            continue
        test   = fitted.test_causality(caused=caused, causing=causing, kind="f")
        fstat  = float(test.test_statistic)
        pval   = float(test.pvalue)
        sig    = pval < ALPHA
        stars  = ("***" if pval < 0.01 else
                  "**"  if pval < 0.05 else
                  "*"   if pval < 0.10 else "")
        label  = "CAUSES ✓" if sig else "no effect"
        print(f"  {causing:6s} → {caused:6s}:  "
              f"F={fstat:7.3f}  p={pval:.4f}{stars:3s}  {label}")
        rows.append({
            "causing":    causing,
            "caused":     caused,
            "direction":  f"{causing}→{caused}",
            "F_stat":     round(fstat, 4),
            "p_value":    round(pval,  4),
            "significant":sig,
            "stars":      stars,
        })

results = pd.DataFrame(rows)
results.to_csv("var_results/granger_results.csv", index=False)
print(f"\nSaved → var_results/granger_results.csv")

# ── Summary: focus on crypto → stablecoin directions ──────────────────────────
print("\n" + "=" * 62)
print("Key directions — crypto → stablecoin")
print("=" * 62)
key = results[results["causing"].isin(["BTC","ETH"]) &
              results["caused"].isin(["USDe","USDC"])]
print(key[["direction","F_stat","p_value","stars","significant"]].to_string(index=False))

print("\n" + "=" * 62)
print("Significant causality links (all pairs, p < 0.05)")
print("=" * 62)
sig = results[results["significant"]]
if len(sig):
    print(sig[["direction","F_stat","p_value","stars"]].to_string(index=False))
else:
    print("  None at p < 0.05")
