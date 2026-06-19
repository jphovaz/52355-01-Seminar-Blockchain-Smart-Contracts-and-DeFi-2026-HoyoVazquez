"""
Robustness Check 3 — Rolling Window CoVaR
==========================================
Re-estimates ΔCoVaR over a rolling window to show time variation.
Plots saved to covar_results/plots/

Input : merged_logreturns.csv
Output: covar_results/rolling_deltacovar.csv
        covar_results/plots/rolling_deltacovar_BTC.png
        covar_results/plots/rolling_deltacovar_ETH.png

Run: python robustness_rolling.py
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
import os

os.makedirs("covar_results/plots", exist_ok=True)

lr = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
lr = lr[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
lr.columns = ["BTC","ETH","USDe","USDC"]

WINDOW = 120    # rolling window in days (approx 4 months)
Q_DIST = 0.05
Q_MED  = 0.50
PAIRS  = [("BTC","USDe"),("BTC","USDC"),("ETH","USDe"),("ETH","USDC")]

print(f"N={len(lr)}  |  Window={WINDOW} days  |  "
      f"Rolling estimates: {len(lr)-WINDOW}")

results = {f"{c}_{s}": [] for c,s in PAIRS}
dates   = []

for i in range(WINDOW, len(lr)):
    window = lr.iloc[i-WINDOW:i]
    dates.append(lr.index[i])

    for crypto, stable in PAIRS:
        x = window[crypto].values
        y = window[stable].values
        X = np.column_stack([np.ones(WINDOW), x])
        try:
            r_d = QuantReg(y, X).fit(q=Q_DIST, vcov="iid")
            r_m = QuantReg(y, X).fit(q=Q_MED,  vcov="iid")
            cd  = r_d.params[0] + r_d.params[1] * x
            cm  = r_m.params[0] + r_m.params[1] * x
            results[f"{crypto}_{stable}"].append(float(np.mean(cd - cm)))
        except Exception:
            results[f"{crypto}_{stable}"].append(np.nan)

roll_df = pd.DataFrame(results, index=dates)
roll_df.index.name = "date"
roll_df.to_csv("covar_results/rolling_deltacovar.csv")
print("Saved → covar_results/rolling_deltacovar.csv")

# ── Plot per crypto ────────────────────────────────────────────────────────────
COLORS = {"USDe": "#D85A30", "USDC": "#378ADD"}

for crypto in ["BTC","ETH"]:
    fig, ax = plt.subplots(figsize=(11, 4))

    for stable in ["USDe","USDC"]:
        series = roll_df[f"{crypto}_{stable}"]
        ax.plot(series.index, series.values,
                color=COLORS[stable], lw=1.2, label=f"ΔCoVaR {crypto}→{stable}")
        ax.fill_between(series.index, series.values, 0,
                        where=(series.values < 0),
                        color=COLORS[stable], alpha=0.12)

    ax.axhline(0, color="grey", lw=0.7, linestyle="--")
    ax.set_title(f"Rolling ΔCoVaR ({WINDOW}-day window)  |  "
                 f"{crypto} stress → USDe vs USDC peg deviation",
                 fontsize=10)
    ax.set_ylabel("ΔCoVaR")
    ax.set_xlabel("date")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = f"covar_results/plots/rolling_deltacovar_{crypto}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

