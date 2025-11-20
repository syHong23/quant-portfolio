"""
Portfolio Optimization - Markowitz Model
Author: Sungyeon Hong
Description: Implements mean-variance portfolio optimization with efficient frontier analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" PORTFOLIO OPTIMIZATION - MARKOWITZ MODEL")
print("="*70)

# Define portfolio assets
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
print(f"\nDownloading data for: {', '.join(tickers)}...")

# Download historical data
data = {}
for ticker in tickers:
    tick = yf.Ticker(ticker)
    hist = tick.history(start='2022-01-01', end='2024-11-01')
    data[ticker] = hist['Close']

prices = pd.DataFrame(data)
returns = prices.pct_change().dropna()
print("Data downloaded successfully")

# Calculate expected returns and covariance matrix
mean_returns = returns.mean() * 252  # Annualized
cov_matrix = returns.cov() * 252     # Annualized

def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calculate portfolio return and volatility"""
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std

def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Calculate negative Sharpe ratio for minimization"""
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std

# Optimization constraints and bounds
num_assets = len(tickers)
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
bounds = tuple((0, 1) for _ in range(num_assets))
initial_guess = num_assets * [1. / num_assets]

print("\nOptimizing portfolio...")
result = minimize(negative_sharpe, initial_guess,
                  args=(mean_returns, cov_matrix),
                  method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result.x

# Display optimal allocation
print(f"\nOPTIMAL PORTFOLIO ALLOCATION:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"   {ticker:6s}: {weight*100:>6.2f}%")

# Calculate optimal portfolio metrics
opt_return, opt_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
opt_sharpe = (opt_return - 0.02) / opt_std

print(f"\nPERFORMANCE METRICS:")
print(f"   Expected Annual Return: {opt_return*100:>6.2f}%")
print(f"   Annual Volatility:      {opt_std*100:>6.2f}%")
print(f"   Sharpe Ratio:           {opt_sharpe:>6.2f}")

# Generate efficient frontier using Monte Carlo simulation
print("\nGenerating Efficient Frontier (10,000 random portfolios)...")
np.random.seed(42)
portfolio_returns = []
portfolio_volatilities = []
portfolio_sharpes = []

for _ in range(10000):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    p_sharpe = (p_return - 0.02) / p_std
    portfolio_returns.append(p_return)
    portfolio_volatilities.append(p_std)
    portfolio_sharpes.append(p_sharpe)

# Visualization
plt.figure(figsize=(14, 9))
scatter = plt.scatter(portfolio_volatilities, portfolio_returns, 
                     c=portfolio_sharpes, cmap='viridis', 
                     alpha=0.6, s=15, edgecolors='none')
plt.colorbar(scatter, label='Sharpe Ratio')
plt.scatter(opt_std, opt_return, c='red', s=500, marker='*', 
            edgecolors='black', linewidths=2, 
            label=f'Optimal Portfolio (Sharpe: {opt_sharpe:.2f})', zorder=5)

plt.title('Efficient Frontier - Portfolio Optimization', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
plt.ylabel('Expected Return', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('portfolio_optimization_results.png', dpi=300, bbox_inches='tight')
print("Chart saved: portfolio_optimization_results.png")
print("="*70)
