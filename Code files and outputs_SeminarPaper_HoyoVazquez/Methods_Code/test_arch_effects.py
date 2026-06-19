"""
ARCH Effects Test — all 4 assets
=================================
Tests whether GARCH modelling is justified for each residual series.

Two complementary tests per asset:

  1. Ljung-Box on squared residuals  ε²_t
     Tests for lag autocorrelation in the variance process directly.
     H0: no autocorrelation in ε²_t-n (no volatility clustering)
     Significant → variance is time-varying → GARCH warranted

  2. Engle (1982) ARCH-LM test
     Regresses ε²_t on its own lags and tests joint significance (F-test).
     H0: no ARCH effects (all lagged ε² coefficients = 0)
     Significant → conditional heteroskedasticity present → GARCH warranted

Both tests should agree. If either is significant, GARCH is justified.

Input : garch_results/merged_residuals.csv
Output: garch_results/arch_test_results.csv   — consolidated results table
        garch_results/arch_test_summary.txt   — plain-text interpretation
        garch_results/squared_resid_plot.png  — visual of ε²_t per asset

Run: python test_arch_effects.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import os

INPUT_FILE = "garch_results/merged_residuals.csv"
OUTPUT_DIR = "garch_results"
LB_LAGS    = [5, 10, 20]   # standard lags for variance clustering check
ARCH_LAGS  = 10            # lags for ARCH-LM test (Engle 1982)
ALPHA      = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE, index_col="date", parse_dates=["date"])
print(f"Loaded: {len(df)} rows  |  columns: {list(df.columns)}\n")

results_rows = []
summary_lines = []

# ── Plot setup ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

# ══════════════════════════════════════════════════════════════════════════════
for i, col in enumerate(df.columns):
    asset = col.replace("resid_", "")
    e     = df[col].dropna()
    e2    = e ** 2                    # squared residuals — proxy for variance

    print("=" * 58)
    print(f"  {asset}")
    print("=" * 58)

    # ── Test 1: Ljung-Box on ε²_t ──────────────────────────────────────────
    lb = acorr_ljungbox(e2, lags=LB_LAGS, return_df=True)
    lb_sig = (lb["lb_pvalue"] < ALPHA).any()
    lb_min_p = float(lb["lb_pvalue"].min())

    print(f"\n  Ljung-Box on ε²_t  (lags {LB_LAGS}):")
    for _, row in lb.iterrows():
        flag = "✓ ARCH" if row["lb_pvalue"] < ALPHA else "—"
        print(f"    lag {int(row.name):2d}: Q={row['lb_stat']:7.3f}  "
              f"p={row['lb_pvalue']:.4f}  {flag}")

    # ── Test 2: Engle ARCH-LM ─────────────────────────────────────────────
    lm_stat, lm_pval, f_stat, f_pval = het_arch(e, nlags=ARCH_LAGS)
    arch_sig = f_pval < ALPHA

    print(f"\n  ARCH-LM test  (lags={ARCH_LAGS}):")
    print(f"    LM stat = {lm_stat:.4f}   LM p = {lm_pval:.4f}")
    print(f"    F  stat = {f_stat:.4f}    F  p = {f_pval:.4f}  "
          f"{'✓ ARCH effects' if arch_sig else '— no ARCH'}")

    # ── Decision ──────────────────────────────────────────────────────────
    garch_justified = lb_sig or arch_sig
    verdict = "GARCH JUSTIFIED ✓" if garch_justified else "no ARCH — GARCH may not add value"
    print(f"\n  Decision: {verdict}\n")

    # ── Squared residuals plot ─────────────────────────────────────────────
    ax = axes[i]
    ax.plot(e2.index, e2.values, lw=0.7,
            color=["#D85A30","#378ADD","#9B59B6","#1D9E75"][i])
    ax.set_title(f"{asset} — squared residuals ε²_t  |  {verdict}",
                 fontsize=9)
    ax.set_ylabel("ε²")

    results_rows.append({
        "asset":              asset,
        "n":                  len(e),
        "mean_sq_resid":      round(float(e2.mean()), 8),
        "lb_lag10_stat":      round(float(lb.loc[10, "lb_stat"]) if 10 in lb.index
                                    else float(lb["lb_stat"].iloc[-1]), 4),
        "lb_lag10_pval":      round(float(lb.loc[10, "lb_pvalue"]) if 10 in lb.index
                                    else float(lb["lb_pvalue"].iloc[-1]), 4),
        "lb_significant":     lb_sig,
        "archlm_f_stat":      round(f_stat, 4),
        "archlm_f_pval":      round(f_pval, 4),
        "archlm_significant": arch_sig,
        "garch_justified":    garch_justified,
        "verdict":            verdict,
    })

    summary_lines.append(
        f"{asset:6s}: LB(ε²) p={lb_min_p:.4f} {'*' if lb_sig else ' '} | "
        f"ARCH-LM F-p={f_pval:.4f} {'*' if arch_sig else ' '} | "
        f"{verdict}"
    )

# ── Save plot ──────────────────────────────────────────────────────────────────
plt.suptitle("Squared Residuals ε²_t — Volatility Clustering Diagnostic",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plot_path = f"{OUTPUT_DIR}/squared_resid_plot.png"
plt.savefig(plot_path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved → {plot_path}")

# ── Save results table ─────────────────────────────────────────────────────────
results_df = pd.DataFrame(results_rows)
csv_path   = f"{OUTPUT_DIR}/arch_test_results.csv"
results_df.to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")

# ── Save plain-text summary ────────────────────────────────────────────────────
txt_path = f"{OUTPUT_DIR}/arch_test_summary.txt"
with open(txt_path, "w") as f:
    f.write("ARCH Effects Test Summary\n")
    f.write("=" * 58 + "\n")
    for line in summary_lines:
        f.write(line + "\n")
    f.write("\n* = significant at 5% level\n")
    f.write("\nInterpretation:\n")
    f.write("  Ljung-Box on ε²_t: tests autocorrelation in squared residuals\n")
    f.write("  ARCH-LM F-test   : Engle (1982) test for conditional heteroskedasticity\n")
    f.write("  Both significant  → strong evidence for GARCH\n")
    f.write("  One significant   → GARCH still warranted\n")
    f.write("  Neither           → constant variance; GARCH adds little\n")
print(f"Saved → {txt_path}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("ARCH TEST SUMMARY")
print("=" * 58)
for line in summary_lines:
    print(" ", line)
print("\n  * = significant at 5% level")
print(f"\n  Full results → {csv_path}")
