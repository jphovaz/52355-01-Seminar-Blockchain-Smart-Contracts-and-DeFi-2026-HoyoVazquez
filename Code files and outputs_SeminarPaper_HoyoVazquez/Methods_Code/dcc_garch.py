"""
DCC-GARCH Model
===============
Dynamic Conditional Correlation GARCH (Engle 2002)

Two-stage estimation:
  Stage 1: Univariate GJR-GARCH(1,1) per asset → standardised residuals z_t
  Stage 2: DCC model on z_t → time-varying correlation matrix R_t

Assets: BTC, ETH, USDe, USDC  (all log returns)

Outputs:
  dcc_results/
    h_t.csv                Individual conditional variances (from stage 1)
    sigma_t.csv            Individual conditional volatilities √h_t
    z_t.csv                Standardised residuals
    dcc_correlations.csv   Time-varying pairwise correlations (upper triangle)
    dcc_params.csv         DCC parameters a and b
    H_t/                   Full 4×4 conditional covariance matrix per day
                           (saved as one CSV: date + 16 elements)

Input : merged_logreturns.csv

Install: pip install arch scipy
Run    : python dcc_garch.py
"""

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings("ignore")
os.makedirs("dcc_results/H_t", exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("Methodology/merged_logreturns.csv", index_col="date", parse_dates=["date"])
df = df[["logret_BTC","logret_ETH","logret_USDe","logret_USDC"]].dropna()
df.columns = ["BTC","ETH","USDe","USDC"]

ASSETS = list(df.columns)
T      = len(df)
K      = len(ASSETS)
print(f"Data: {T} observations, {K} assets: {ASSETS}")
print(f"Sample: {df.index[0].date()} → {df.index[-1].date()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Univariate GJR-GARCH(1,1) per asset
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 58)
print("Stage 1: Univariate GJR-GARCH(1,1) per asset")
print("=" * 58)

h_t     = pd.DataFrame(index=df.index, columns=ASSETS, dtype=float)
sigma_t = pd.DataFrame(index=df.index, columns=ASSETS, dtype=float)
z_t     = pd.DataFrame(index=df.index, columns=ASSETS, dtype=float)
stage1_params = []

for asset in ASSETS:
    r = df[asset].values * 100   # scale to % for numerical stability

    model  = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
    result = model.fit(disp="off", show_warning=False)

    # Extract and rescale back to decimal units
    cv        = result.conditional_volatility / 100      # σ_t in decimal
    h         = cv ** 2                                  # h_t = σ²_t
    z         = result.std_resid                         # standardised residuals

    h_t[asset]     = h
    sigma_t[asset] = cv
    z_t[asset]     = z

    params = result.params
    persist = float(params["alpha[1]"]) + float(params["beta[1]"]) + \
              0.5 * float(params["gamma[1]"])

    print(f"\n  {asset}")
    print(f"    ω={float(params['omega'])/1e4:.6f}  "
          f"α={float(params['alpha[1]']):.4f}  "
          f"γ={float(params['gamma[1]']):.4f}  "
          f"β={float(params['beta[1]']):.4f}  "
          f"ν={float(params['nu']):.2f}")
    print(f"    Persistence={persist:.4f}")
    stage1_params.append({
        "asset": asset,
        "omega": float(params["omega"]) / 1e4,
        "alpha": float(params["alpha[1]"]),
        "gamma": float(params["gamma[1]"]),
        "beta":  float(params["beta[1]"]),
        "nu":    float(params["nu"]),
        "persistence": persist,
    })

# Save stage 1 outputs
h_t.to_csv("dcc_results/h_t.csv")
sigma_t.to_csv("dcc_results/sigma_t.csv")
z_t.to_csv("dcc_results/z_t.csv")
pd.DataFrame(stage1_params).to_csv("dcc_results/stage1_params.csv", index=False)
print(f"\nStage 1 outputs saved.")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DCC(1,1) on standardised residuals
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("Stage 2: DCC(1,1) — dynamic conditional correlation")
print("=" * 58)

"""
DCC model (Engle 2002):

  Q_t = (1 - a - b) * Q̄  +  a * z_{t-1} z'_{t-1}  +  b * Q_{t-1}

  R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}

where:
  Q̄   = unconditional covariance matrix of z_t (K×K, estimated from data)
  Q_t = pseudo-correlation matrix at time t
  R_t = standardised correlation matrix at time t (diagonal elements = 1)
  a   = ARCH effect in correlations (how fast correlations react to shocks)
  b   = GARCH effect in correlations (how persistent correlations are)
  a + b < 1 required for stationarity

The conditional covariance matrix is then:
  H_t = D_t * R_t * D_t
where D_t = diag(σ_BTC_t, σ_ETH_t, σ_USDe_t, σ_USDC_t)
"""

Z = z_t.values   # T × K matrix of standardised residuals
Q_bar = np.cov(Z.T)   # unconditional covariance of z_t (K×K)

def dcc_loglik(params):
    """
    Negative log-likelihood of DCC model given parameters [a, b].
    Used for numerical optimisation.
    """
    a, b = params
    if a <= 0 or b <= 0 or a + b >= 1:
        return 1e10   # return large value for invalid parameters

    Q  = Q_bar.copy()
    ll = 0.0

    for t in range(1, T):
        z   = Z[t-1, :]
        Q   = (1 - a - b) * Q_bar + a * np.outer(z, z) + b * Q

        # Standardise Q → R
        d_inv = 1.0 / np.sqrt(np.diag(Q))
        R     = Q * np.outer(d_inv, d_inv)

        # Log-likelihood contribution at t
        # ll += -0.5 * (log|R| + z'_t R^{-1} z_t)
        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        z_t_vec = Z[t, :]
        try:
            R_inv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            return 1e10
        ll += -0.5 * (logdet + z_t_vec @ R_inv @ z_t_vec)

    return -ll   # return negative (minimising)


print("  Optimising DCC parameters a and b...")
result_dcc = minimize(
    dcc_loglik,
    x0     = [0.05, 0.90],             # starting values (standard in literature)
    bounds = [(1e-6, 0.5), (1e-6, 0.999)],
    method = "L-BFGS-B",
)

a_opt, b_opt = result_dcc.x
print(f"  DCC parameters: a = {a_opt:.6f}  b = {b_opt:.6f}")
print(f"  a + b = {a_opt + b_opt:.6f}  "
      f"({'stationary ✓' if a_opt + b_opt < 1 else 'non-stationary ✗'})")
print(f"  Optimisation: {'converged ✓' if result_dcc.success else 'did not converge — check starting values'}")

pd.DataFrame([{
    "a": round(a_opt, 6), "b": round(b_opt, 6),
    "a_plus_b": round(a_opt + b_opt, 6),
    "converged": result_dcc.success,
}]).to_csv("dcc_results/dcc_params.csv", index=False)


# ── Extract time-varying correlations and covariance matrices ──────────────────
print("\n  Extracting time-varying R_t and H_t...")

pairs = [(i, j) for i in range(K) for j in range(i+1, K)]
pair_labels = [f"corr_{ASSETS[i]}_{ASSETS[j]}" for i,j in pairs]

corr_df  = pd.DataFrame(index=df.index, columns=pair_labels, dtype=float)
H_t_rows = []   # full covariance matrix flattened per day

Q = Q_bar.copy()
D_t = sigma_t.values   # T × K

for t in range(1, T):
    z = Z[t-1, :]
    Q = (1 - a_opt - b_opt) * Q_bar + a_opt * np.outer(z, z) + b_opt * Q

    # Correlation matrix R_t
    d_inv = 1.0 / np.sqrt(np.diag(Q))
    R     = Q * np.outer(d_inv, d_inv)

    # Store pairwise correlations
    for (i, j), label in zip(pairs, pair_labels):
        corr_df.loc[df.index[t], label] = R[i, j]

    # Full conditional covariance matrix H_t = D_t R_t D_t
    D     = np.diag(D_t[t, :])
    H     = D @ R @ D

    row = {"date": df.index[t]}
    for i, ai in enumerate(ASSETS):
        for j, aj in enumerate(ASSETS):
            row[f"H_{ai}_{aj}"] = H[i, j]
    H_t_rows.append(row)

corr_df = corr_df.dropna()
H_t_df  = pd.DataFrame(H_t_rows).set_index("date")

corr_df.to_csv("dcc_results/dcc_correlations.csv")
H_t_df.to_csv("dcc_results/H_t_matrix.csv")

print(f"  Time-varying correlations saved → dcc_results/dcc_correlations.csv")
print(f"  Full H_t matrices saved         → dcc_results/H_t_matrix.csv")


# ── Summary statistics on dynamic correlations ─────────────────────────────────
print("\n" + "=" * 58)
print("Dynamic correlation summary (mean ± std over sample)")
print("=" * 58)
print(corr_df.describe().loc[["mean","std","min","max"]].round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 58)
print("Complete")
print("=" * 58)
print(f"""
Outputs:
  dcc_results/stage1_params.csv     GJR-GARCH parameters per asset
  dcc_results/h_t.csv               Conditional variances h_t (stage 1)
  dcc_results/sigma_t.csv           Conditional volatilities σ_t
  dcc_results/z_t.csv               Standardised residuals
  dcc_results/dcc_params.csv        DCC parameters a and b
  dcc_results/dcc_correlations.csv  Time-varying pairwise correlations
  dcc_results/H_t_matrix.csv        Full 4×4 conditional covariance per day

Next steps using these outputs:
  Granger on volatility → use h_t.csv as the input series 
  Diebold-Yilmaz        → uses H_t_matrix.csv for spillover decomposition
      
      --> OUTPUTS NOT USED DUE TO NEAR-ZERO VOLATILITY IN STABLECOIN USDC YIELDING DEGENERATE COVARIANCE MATRICES.s
""")
