"""
Granger Causality on Volatility — diagnostic + squared returns proxy
====================================================================
Problem with h_t Granger: stablecoin h_t is near-constant → low power.
Solution: use squared log returns r2_t as a model-free realized volatility
proxy, which has more variation and is standard in the spillover literature.

Two approaches compared:
  A) VAR Granger on h_t     (GARCH-based, smooth)
  B) VAR Granger on r2_t    (model-free, more variation)

Input : merged_logreturns.csv
        dcc_results/h_t.csv
Output: dcc_results/granger_volatility_ht.csv
        dcc_results/granger_volatility_r2.csv
        dcc_results/volatility_diagnostics.csv

Run: python granger_volatility.py (Granger in squared returns and GARCH conditional variance h_t for comparison)
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import os

os.makedirs("dcc_results", exist_ok=True)

ALPHA    = 0.05
MAX_LAGS = 10

# Load
lr = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
lr = lr[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
lr.columns = ["BTC","ETH","USDe","USDC"]

h = pd.read_csv("dcc_results/h_t.csv", index_col="date", parse_dates=["date"])
h = h.dropna()

# Squared returns
r2 = lr ** 2
r2.columns = ["BTC","ETH","USDe","USDC"]

# Step 1: Variation diagnostic
print("=" * 62)
print("Step 1: Variation diagnostic")
print("=" * 62)
cv_ht = (h.std()  / h.mean()).rename("CV_h_t")
cv_r2 = (r2.std() / r2.mean()).rename("CV_r2_t")
diag  = pd.concat([h.mean().rename("mean_h"), cv_ht,
                   r2.mean().rename("mean_r2"), cv_r2], axis=1).round(8)
print(diag.to_string())
diag.to_csv("dcc_results/volatility_diagnostics.csv")
print("\nIf CV stablecoin h_t is much smaller than CV crypto h_t,")
print("the Granger test on h_t has low power for stablecoins.")


def run_granger(data, label, outfile):
    assets   = list(data.columns)
    lag_rows = []
    for p in range(1, MAX_LAGS + 1):
        try:
            res = VAR(data).fit(maxlags=p, ic=None, trend="c")
            lag_rows.append({"lag": p, "AIC": res.aic, "BIC": res.bic})
        except Exception:
            continue
    lag_df = pd.DataFrame(lag_rows).set_index("lag")
    P      = int(lag_df["BIC"].idxmin())
    fitted = VAR(data).fit(maxlags=P, ic=None, trend="c")

    print("\n" + "=" * 62)
    print(f"Granger — {label}  [VAR({P})]")
    print("=" * 62)

    rows = []
    for caused in assets:
        for causing in assets:
            if caused == causing:
                continue
            test  = fitted.test_causality(caused=caused, causing=causing, kind="f")
            fstat = float(test.test_statistic)
            pval  = float(test.pvalue)
            sig   = pval < ALPHA
            stars = "***" if pval<0.01 else "**" if pval<0.05 else "*" if pval<0.10 else ""
            print(f"  {causing:6s} → {caused:6s}: "
                  f"F={fstat:7.3f}  p={pval:.4f}{stars:3s}  "
                  f"{'CAUSES' if sig else 'no effect'}")
            rows.append({"causing":causing,"caused":caused,
                         "direction":f"{causing}→{caused}",
                         "F_stat":round(fstat,4),"p_value":round(pval,4),
                         "significant":sig,"stars":stars})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(outfile, index=False)
    print(f"Saved → {outfile}")
    return df_out


res_ht = run_granger(h,                            "h_t (GARCH conditional variance)",
                     "dcc_results/granger_volatility_ht.csv")
res_r2 = run_granger(r2.reindex(h.index).dropna(), "r2_t (squared returns proxy)",
                     "dcc_results/granger_volatility_r2.csv")

# Comparison table — crypto → stablecoin only
print("\n" + "=" * 62)
print("Comparison: crypto → stablecoin  (h_t vs r2_t)")
print("=" * 62)

def key_dirs(df):
    return df[df["causing"].isin(["BTC","ETH"]) &
              df["caused"].isin(["USDe","USDC"])][
        ["direction","p_value","significant"]]

comp = key_dirs(res_ht).rename(columns={"p_value":"p_ht","significant":"sig_ht"}).merge(
       key_dirs(res_r2).rename(columns={"p_value":"p_r2","significant":"sig_r2"}),
       on="direction")
print(comp.to_string(index=False))
print("""
sig_ht=F, sig_r2=T  → volatility channel exists but h_t lacks power
                       use r2_t as primary, h_t as robustness
Both False           → no linear volatility Granger; CoVaR is primary evidence
Both True            → strong evidence across both proxies
""")

"""
Results:
==============================================================
Comparison: crypto → stablecoin  (h_t vs r2_t)
==============================================================
direction   p_ht  sig_ht   p_r2  sig_r2
 BTC→USDe 0.2538   False 0.0197    True
 ETH→USDe 0.6336   False 0.2208   False
 BTC→USDC 0.6697   False 0.6490   False
 ETH→USDC 0.7055   False 0.7216   False
"""