#Time Series Forecasting for Portfolio Management Optimization

## Overview
This project is part of a training challenge for **GMF Investments**.  
It applies **time series forecasting** to historical financial data from three key assets:

- **TSLA** – High-growth, high-volatility stock (Tesla)
- **BND** – Bond ETF for stability
- **SPY** – S&P 500 ETF for diversified exposure

The goal is to forecast market movements, optimize portfolio allocation, and evaluate risk-adjusted performance to help achieve better investment decisions.


## Business Context
Predicting exact stock prices is extremely challenging due to the **Efficient Market Hypothesis (EMH)**.  
Instead, models are used to:
- Forecast volatility
- Detect trends and momentum factors
- Support a **multi-factor decision-making framework** for portfolio optimization


## Objectives
1. **Preprocess & Explore Data**
   - Fetch historical data (2015-07-01 → 2025-07-31) using `yfinance`
   - Clean, scale, and handle missing values
   - Conduct **Exploratory Data Analysis (EDA)**:
     - Price trends
     - Daily returns
     - Rolling volatility
     - Value at Risk (VaR) & Sharpe Ratio
     - Augmented Dickey-Fuller (ADF) test for stationarity

2. **Build & Compare Forecasting Models**
   - **ARIMA/SARIMA** – Classical statistical forecasting
   - **LSTM** – Deep learning-based time series prediction
   - Evaluate using RMSE, MAE, and MAPE
   - Chronological train-test split (e.g., 2015–2023 train, 2024–2025 test)

3. **Portfolio Optimization**
   - Use `PyPortfolioOpt` to find the **Efficient Frontier**
   - Simulate portfolio performance via backtesting
   - Recommend allocation strategies balancing risk & return


## Tech Stack
- **Python**: Data analysis & modeling
- **Libraries**:
  - Data: `pandas`, `numpy`, `yfinance`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
  - Statistical Modeling: `statsmodels`, `pmdarima`
  - Deep Learning: `tensorflow` / `keras`
  - Portfolio Optimization: `PyPortfolioOpt`
- **Jupyter Notebook** for interactive analysis
- **.py scripts** for reproducibility

 
