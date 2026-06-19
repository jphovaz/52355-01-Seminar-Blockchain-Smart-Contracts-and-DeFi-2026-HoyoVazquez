"""
Diebold-Yilmaz Spillover Index — squared returns r2_t
Generalised FEVD computed manually (avoids statsmodels API inconsistencies)
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import os

os.makedirs("dcc_results", exist_ok=True)

H        = 10    # forecast horizon
MAX_LAGS = 10

# Load and build r2
lr = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
lr = lr[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
lr.columns = ["BTC","ETH","USDe","USDC"]
r2 = lr ** 2
assets = list(r2.columns)
K = len(assets)

# Lag selection
rows = []
for p in range(1, MAX_LAGS + 1):
    try:
        res = VAR(r2).fit(maxlags=p, ic=None, trend="c")
        rows.append({"lag": p, "BIC": res.bic})
    except Exception:
        continue
lag_df = pd.DataFrame(rows).set_index("lag")
P = int(lag_df["BIC"].idxmin())
print(f"VAR lag (BIC): p = {P}")

fitted = VAR(r2).fit(maxlags=P, ic=None, trend="c")
Sigma  = fitted.sigma_u          # K x K residual covariance
A      = fitted.coefs            # shape (P, K, K) — coefficient matrices

# Build companion matrix (KP x KP)
KP = K * P
companion = np.zeros((KP, KP))
for i in range(P):
    companion[:K, i*K:(i+1)*K] = A[i]
companion[K:, :K*(P-1)] = np.eye(K * (P-1))

# Compute moving-average coefficients Phi_h via companion matrix
# Phi_h = J @ companion^h @ J'  where J selects first K rows
J    = np.zeros((K, KP)); J[:, :K] = np.eye(K)
Phi  = [J @ np.linalg.matrix_power(companion, h) @ J.T for h in range(H+1)]

# Generalised FEVD (Pesaran-Shin 1998, Diebold-Yilmaz 2012)
# theta[i,j] = sigma_jj^{-1} * sum_h (e_i Phi_h Sigma e_j)^2
#              / sum_h (e_i Phi_h Sigma Phi_h' e_i)
sigma_diag = np.diag(Sigma)
theta = np.zeros((K, K))
for i in range(K):
    ei = np.zeros(K); ei[i] = 1
    denom = sum(float(ei @ Phi[h] @ Sigma @ Phi[h].T @ ei) for h in range(H+1))
    for j in range(K):
        ej = np.zeros(K); ej[j] = 1
        numer = sum(float(ei @ Phi[h] @ Sigma @ ej)**2 for h in range(H+1))
        theta[i, j] = numer / (sigma_diag[j] * denom)

# Row-normalise so shares sum to 1 per row
theta_norm = theta / theta.sum(axis=1, keepdims=True)

# Build spillover table
table = pd.DataFrame(theta_norm * 100, index=assets, columns=assets)
table["FROM"] = [table.iloc[i].sum() - table.iloc[i, i] for i in range(K)]
to_row = pd.Series({a: table[a].sum() - table.loc[a, a] for a in assets}, name="TO")
net_row= pd.Series({a: to_row[a] - table.loc[a,"FROM"] for a in assets}, name="NET")
table.loc["TO"]  = to_row
table.loc["NET"] = net_row

total = float(to_row.sum() / K)
table.loc["Total%"] = total

print("\nDiebold-Yilmaz Spillover Table (%)  H =", H)
print("Rows = forecast error of [row] explained by shocks from [col]")
print(table.round(2).to_string())
print(f"\nTotal Spillover Index: {total:.2f}%")

# Key result
print("\nKey: crypto → stablecoin shares")
for stable in ["USDe","USDC"]:
    for crypto in ["BTC","ETH"]:
        val = theta_norm[assets.index(stable), assets.index(crypto)] * 100
        print(f"  {crypto} → {stable}: {val:.3f}%")

table.to_csv("dcc_results/dy_spillover_table.csv")
print("\nSaved → dcc_results/dy_spillover_table.csv")

"""
Results:
Diebold-Yilmaz Spillover Table (%)  H = 10
Rows = forecast error of [row] explained by shocks from [col]
          BTC    ETH   USDe   USDC   FROM
BTC     67.59  31.64   0.75   0.02  32.41
ETH     32.97  65.98   0.75   0.31  34.02
USDe     2.16   1.24  96.48   0.12   3.52
USDC     0.06   0.46   0.19  99.29   0.71
TO      35.19  33.34   1.69   0.45    NaN
NET      2.77  -0.68  -1.83  -0.27    NaN
Total%  17.67  17.67  17.67  17.67  17.67

Total Spillover Index: 17.67%

Key: crypto → stablecoin shares
  BTC → USDe: 2.162%
  ETH → USDe: 1.242%
  BTC → USDC: 0.058%
  ETH → USDC: 0.464%
""" 