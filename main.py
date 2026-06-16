# portfolio_optimization.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

# ── 1. Synthetic NIFTY50 Price Data (15 stocks, 5 years daily) ────────────────
np.random.seed(42)
STOCKS = ['RELIANCE','TCS','HDFC','INFY','ICICI','WIPRO','AXISBANK',
          'SBIN','LT','BAJFINANCE','HCLTECH','MARUTI','ITC','ASIANPAINT','NESTLEIND']
N_DAYS = 1260  # ~5 years
MARKET_RET = 0.0004  # daily market return

annual_ret  = np.random.uniform(0.08, 0.22, len(STOCKS))
annual_vol  = np.random.uniform(0.18, 0.40, len(STOCKS))
betas       = np.random.uniform(0.6, 1.4, len(STOCKS))

daily_ret  = annual_ret  / 252
daily_vol  = annual_vol  / np.sqrt(252)

# Correlated returns
corr_base = np.full((len(STOCKS), len(STOCKS)), 0.35)
np.fill_diagonal(corr_base, 1.0)
L = np.linalg.cholesky(corr_base)
raw = np.random.randn(N_DAYS, len(STOCKS))
corr_raw = raw @ L.T
returns_mat = corr_raw * daily_vol + daily_ret

prices = pd.DataFrame(
    (np.exp(np.cumsum(returns_mat, axis=0)) * 100),
    columns=STOCKS
)

print(f"Price data shape: {prices.shape}")
print(prices.describe().round(2))

# ── 2. Returns & Covariance ───────────────────────────────────────────────────
returns = prices.pct_change().dropna()
mu      = returns.mean() * 252          # annualized mean returns
cov     = returns.cov()  * 252          # annualized cov matrix
corr_m  = returns.corr()

print(f"\nAnnualized Returns (mean): {mu.mean():.2%}")

# ── 3. Efficient Frontier (10,000 random portfolios) ─────────────────────────
N = len(STOCKS)
n_port = 10_000
port_returns = np.zeros(n_port)
port_vols    = np.zeros(n_port)
port_sharpe  = np.zeros(n_port)
port_weights = np.zeros((n_port, N))
RF = 0.065  # risk-free rate (RBI repo ~6.5%)

for i in range(n_port):
    w = np.random.dirichlet(np.ones(N))
    r = w @ mu.values
    v = np.sqrt(w @ cov.values @ w)
    port_returns[i] = r
    port_vols[i]    = v
    port_sharpe[i]  = (r - RF) / v
    port_weights[i] = w

# ── 4. Optimal Portfolio (Max Sharpe via SciPy) ───────────────────────────────
def neg_sharpe(w, mu, cov, rf=RF):
    r = w @ mu
    v = np.sqrt(w @ cov @ w)
    return -(r - rf) / v

constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
bounds      = [(0.01, 0.40)] * N
w0          = np.ones(N) / N

result = minimize(neg_sharpe, w0, args=(mu.values, cov.values),
                  method='SLSQP', bounds=bounds, constraints=constraints)
opt_w = result.x
opt_r = opt_w @ mu.values
opt_v = np.sqrt(opt_w @ cov.values @ opt_w)
opt_s = (opt_r - RF) / opt_v

print(f"\n🏆 Optimal Portfolio:")
print(f"   Expected Return: {opt_r:.2%}")
print(f"   Volatility:      {opt_v:.2%}")
print(f"   Sharpe Ratio:    {opt_s:.4f}")

# ── 5. CAPM ──────────────────────────────────────────────────────────────────
market_daily = np.random.randn(N_DAYS - 1) * 0.008 + MARKET_RET
mkt_ann_ret  = (1 + MARKET_RET) ** 252 - 1

capm_betas   = []
capm_alphas  = []
for stock in STOCKS:
    stock_r  = returns[stock].values
    cov_sm   = np.cov(stock_r, market_daily)[0, 1]
    var_m    = np.var(market_daily)
    beta     = cov_sm / var_m
    expected = RF + beta * (mkt_ann_ret - RF)
    alpha    = mu[stock] - expected
    capm_betas.append(beta)
    capm_alphas.append(alpha)

capm_df = pd.DataFrame({'Stock': STOCKS, 'Beta': capm_betas, 'Alpha': capm_alphas})
print("\n📊 CAPM Results:")
print(capm_df.round(4).to_string(index=False))

# ── 6. Visualizations ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Portfolio Optimization — Markowitz & CAPM', fontsize=16, fontweight='bold')

# Efficient Frontier
sc = axes[0,0].scatter(port_vols, port_returns, c=port_sharpe,
                        cmap='viridis', alpha=0.4, s=5)
axes[0,0].scatter(opt_v, opt_r, c='red', s=200, zorder=5,
                   marker='*', label=f'Max Sharpe ({opt_s:.2f})')
plt.colorbar(sc, ax=axes[0,0], label='Sharpe Ratio')

# Capital Market Line
x_cml = np.linspace(0, port_vols.max(), 200)
y_cml = RF + opt_s * x_cml
axes[0,0].plot(x_cml, y_cml, 'r--', linewidth=1.5, label='CML')
axes[0,0].set_xlabel('Volatility'); axes[0,0].set_ylabel('Return')
axes[0,0].set_title('Efficient Frontier & CML'); axes[0,0].legend()

# Optimal Weights
axes[0,1].barh(STOCKS, opt_w * 100, color='steelblue', edgecolor='k')
axes[0,1].set_xlabel('Weight (%)')
axes[0,1].set_title('Optimal Portfolio Allocation')

# CAPM Security Market Line
sml_x = np.linspace(0, 2, 100)
sml_y = RF + sml_x * (mkt_ann_ret - RF)
axes[1,0].plot(sml_x, sml_y, 'r-', label='SML')
axes[1,0].scatter(capm_betas, mu.values, color='steelblue', zorder=5)
for i, s in enumerate(STOCKS):
    axes[1,0].annotate(s, (capm_betas[i], mu.values[i]), fontsize=7, alpha=0.8)
axes[1,0].set_xlabel('Beta'); axes[1,0].set_ylabel('Expected Return')
axes[1,0].set_title('CAPM — Security Market Line'); axes[1,0].legend()

# Correlation Heatmap
mask = np.triu(np.ones_like(corr_m, dtype=bool))
sns_data = corr_m.values
im = axes[1,1].imshow(sns_data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
axes[1,1].set_xticks(range(N)); axes[1,1].set_xticklabels(STOCKS, rotation=90, fontsize=7)
axes[1,1].set_yticks(range(N)); axes[1,1].set_yticklabels(STOCKS, fontsize=7)
plt.colorbar(im, ax=axes[1,1]); axes[1,1].set_title('Correlation Matrix')

plt.tight_layout()
plt.savefig('portfolio_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved: portfolio_results.png")