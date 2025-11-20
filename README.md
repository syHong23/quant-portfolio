# Portfolio Optimization - Markowitz Model

An investment portfolio optimizer that helps you find the best way to distribute money across multiple stocks to maximize returns while minimizing risk.

## What Does This Do?

Imagine you have $100,000 to invest in 5 tech stocks (Apple, Microsoft, Google, Amazon, NVIDIA). How much should you put in each stock to get the best return for the risk you're taking?

This project uses the **Markowitz Mean-Variance Optimization** model to answer that question. It analyzes 3 years of historical stock data and finds the optimal allocation that gives you the highest risk-adjusted return (Sharpe Ratio).

## Key Concepts Explained Simply

### Sharpe Ratio
Think of this as your "bang for your buck" score. A Sharpe Ratio of 1.18 means for every unit of risk you take, you get 1.18 units of return. Higher is better!

### Efficient Frontier
This is a curve showing all the best possible portfolios. Any portfolio below this curve is inefficient (you could get better returns for the same risk, or lower risk for the same returns).

### Volatility (Risk)
How much your portfolio value jumps around. Higher volatility = more unpredictable, higher risk.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python portfolio_optimization.py
```

The script will:
1. Download 3 years of stock data for AAPL, MSFT, GOOGL, AMZN, NVDA
2. Calculate optimal portfolio weights
3. Generate an Efficient Frontier visualization
4. Save results to `portfolio_optimization_results.png`

## Results

### Optimal Portfolio Allocation
- **AAPL**: 0.00%
- **MSFT**: 0.00%
- **GOOGL**: 0.00%
- **AMZN**: 0.00%
- **NVDA**: 100.00%

### Performance Metrics
- **Expected Annual Return**: 68.22%
- **Annual Volatility**: 56.20%
- **Sharpe Ratio**: 1.18

> **Note**: The optimization concentrated 100% in NVDA because it had the highest historical returns during the analyzed period. In real investing, this would be too risky and you'd want to add constraints for diversification.

## Visualization

The generated chart shows:
- **Scatter Plot**: 10,000 randomly generated portfolios
- **Color**: Sharpe Ratio (yellow = better risk-adjusted returns)
- **Red Star**: The mathematically optimal portfolio
- **Efficient Frontier**: The upper edge of the scatter plot

## How It Works

1. **Download Data**: Gets historical closing prices from Yahoo Finance
2. **Calculate Returns**: Computes daily percentage changes
3. **Statistical Analysis**: Calculates mean returns and covariance matrix
4. **Optimization**: Uses mathematical optimization (scipy) to maximize Sharpe Ratio
5. **Monte Carlo Simulation**: Generates 10,000 random portfolios to map the efficient frontier
6. **Visualization**: Creates a scatter plot showing risk vs. return trade-offs

## Technical Stack

- **Python 3.x**
- **Optimization**: scipy.optimize
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib
- **Market Data**: yfinance (Yahoo Finance API)

## Mathematical Foundation

The optimizer solves:
```
Maximize: (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
Subject to: Sum of weights = 1, All weights >= 0
```

Where:
- Portfolio Return = Weighted average of individual stock returns
- Portfolio Volatility = √(weights^T × Covariance Matrix × weights)
- Risk-Free Rate = 2% (US Treasury rate approximation)

## Real-World Applications

- **Asset Allocation**: Institutional investors use this for pension funds, endowments
- **Risk Management**: Quantifying portfolio risk exposure
- **Robo-Advisors**: Automated investment platforms (Wealthfront, Betterment)
- **Hedge Funds**: Building market-neutral strategies

## Limitations

- Based on historical data (past performance ≠ future results)
- Assumes normal distribution of returns
- Doesn't account for transaction costs
- No constraints on concentration risk in this basic version

## Author

Sungyeon Hong - [LinkedIn](https://www.linkedin.com/in/sungyeon-hong/) | [GitHub](https://github.com/syHong23)

## License

This project is for educational and portfolio demonstration purposes.
