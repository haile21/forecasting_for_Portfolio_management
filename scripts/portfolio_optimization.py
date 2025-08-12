# task4_portfolio_optimization.py
"""
Task 4 - Portfolio Optimization using forecasted TSLA expected return + historical BND & SPY.
Outputs: Efficient frontier plot, weights for Max Sharpe and Min Vol portfolios, summary JSON.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, os, json
from datetime import datetime
plt.style.use('seaborn')

# pip install PyPortfolioOpt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage, sample_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Local preprocessing import (if you prefer live data)
from task1_data_preprocessing import preprocess_data

# CONFIG
RISK_FREE_RATE = 0.03
ANNUAL_DAYS = 252
FORECAST_CSV_ARIMA_6M = "task3_outputs/forecast_arima_6m.csv"
FORECAST_CSV_LSTM_6M = "task3_outputs/forecast_lstm_6m.csv"
OUTPUT_DIR = "task4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load historical returns ----------
data = preprocess_data()  # returns pivot_close etc.
pivot_close = data["pivot_close"].sort_index()
# Use adjusted close pivot if you have it; here pivot_close is Close.
prices = pivot_close[['TSLA','BND','SPY']].dropna()

# daily returns
daily_returns = prices.pct_change().dropna()

# ---------- Compute historical expected returns (annualized) for BND & SPY ----------
hist_mean_daily = daily_returns.mean()  # mean of daily returns (TSLA, BND, SPY)
hist_annual = hist_mean_daily * ANNUAL_DAYS  # simple annualization
print("Historical annualized mean returns (simple):")
print(hist_annual[['BND','SPY']])

# ---------- Derive expected return for TSLA from forecast ----------
last_price = prices['TSLA'].iloc[-1]

def tsla_expected_from_forecast(end_of_horizon_price, horizon_days, method="end"):
    """
    method 'end': use (P_h / P_0 - 1) then annualize by * (ANNUAL_DAYS / horizon_days)
    method 'mean': compute average daily return across horizon then annualize
    """
    if method == "end":
        total_ret = (end_of_horizon_price / last_price) - 1.0
        annualized = (1 + total_ret) ** (ANNUAL_DAYS / horizon_days) - 1
        return annualized
    elif method == "mean":
        # assumes we have daily forecasted prices array
        raise ValueError("Use tsla_expected_from_forecast_mean() with array input")
    else:
        raise ValueError("Unknown method")

def tsla_expected_from_forecast_mean(price_array, horizon_days):
    # compute daily pct changes then average and annualize
    price_array = np.array(price_array)
    daily = price_array[1:] / price_array[:-1] - 1
    avg_daily = np.mean(daily)
    return (1+avg_daily)**ANNUAL_DAYS - 1

# try to load forecast (prefer best model; ARIMA shown as example)
tsla_expected = None
used_forecast_file = None
if os.path.exists(FORECAST_CSV_LSTM_6M):
    # if you want LSTM-based expected return
    df_l = pd.read_csv(FORECAST_CSV_LSTM_6M, parse_dates=['date'], index_col='date')
    # use last mean forecast price
    p_h = df_l['lstm_forecast'].iloc[-1]
    horizon = len(df_l)
    tsla_expected = tsla_expected_from_forecast(p_h, horizon, method="end")
    used_forecast_file = FORECAST_CSV_LSTM_6M
elif os.path.exists(FORECAST_CSV_ARIMA_6M):
    df_a = pd.read_csv(FORECAST_CSV_ARIMA_6M, parse_dates=['date'], index_col='date')
    p_h = df_a['arima_forecast'].iloc[-1]
    horizon = len(df_a)
    tsla_expected = tsla_expected_from_forecast(p_h, horizon, method="end")
    used_forecast_file = FORECAST_CSV_ARIMA_6M
else:
    raise FileNotFoundError("No forecast file found. Run Task 3 first and ensure forecast CSVs exist.")

print(f"Using forecast file: {used_forecast_file}, horizon days: {horizon}")
print(f"Derived TSLA expected annual return (from forecast): {tsla_expected:.4f}")

# ---------- Build expected returns vector ----------
expected_returns = pd.Series({
    "TSLA": tsla_expected,
    "BND": hist_annual['BND'],
    "SPY": hist_annual['SPY']
})

# ---------- Covariance matrix (annualized) ----------
# Use historical daily returns to compute covariance, then annualize by factor 252
cov_daily = daily_returns[['TSLA','BND','SPY']].cov()
cov_annual = cov_daily * ANNUAL_DAYS
print("\nAnnualized covariance matrix:")
print(cov_annual)

# ---------- Optimize (Efficient Frontier) ----------
# Use PyPortfolioOpt for optimization
from pypfopt import EfficientFrontier, risk_models, expected_returns

# We give EfficientFrontier the historical expected returns for BND & SPY and TSLA from forecast
mu = expected_returns
S = cov_annual

# Instantiate EfficientFrontier with our mu and cov
ef = EfficientFrontier(mu, S, weight_bounds=(0,1))  # no shorting
# maximize Sharpe
ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=(0,1))
raw_weights_sharpe = ef_max_sharpe.max_sharpe(risk_free_rate=RISK_FREE_RATE)
w_sharpe = ef_max_sharpe.clean_weights()
perf_sharpe = ef_max_sharpe.portfolio_performance(verbose=True, risk_free_rate=RISK_FREE_RATE)

# minimize volatility
ef_min_vol = EfficientFrontier(mu, S, weight_bounds=(0,1))
raw_weights_minvol = ef_min_vol.min_volatility()
w_minvol = ef_min_vol.clean_weights()
perf_minvol = ef_min_vol.portfolio_performance(verbose=True, risk_free_rate=RISK_FREE_RATE)

# ---------- Efficient frontier (sample points) ----------
from pypfopt import plotting
# generate efficient frontier points
ef_plot = EfficientFrontier(mu, S, weight_bounds=(0,1))
n_points = 50
ret_vals = []
vol_vals = []
weights_list = []
for target in np.linspace(mu.min(), mu.max(), n_points):
    try:
        ef_plot.efficient_return(target_return=target)
        w = ef_plot.clean_weights()
        ret, vol, sharpe = ef_plot.portfolio_performance(risk_free_rate=RISK_FREE_RATE)
        ret_vals.append(ret)
        vol_vals.append(vol)
        weights_list.append(w)
    except Exception:
        continue

# Plot frontier
plt.figure(figsize=(10,6))
plt.plot(vol_vals, ret_vals, 'b--', label='Efficient Frontier')
# Plot individual assets
for asset in mu.index:
    plt.scatter(np.sqrt(S.loc[asset, asset]), mu[asset], marker='o', s=80, label=asset)
# mark min vol and max sharpe
# compute portfolios' performance
def portfolio_metrics(weights, mu, S, rf=RISK_FREE_RATE):
    w = np.array([weights[a] for a in mu.index])
    port_ret = np.dot(w, mu.values)
    port_vol = np.sqrt(w.T.dot(S.values).dot(w))
    port_sharpe = (port_ret - rf)/port_vol
    return port_ret, port_vol, port_sharpe

ret_sh, vol_sh, sr_sh = portfolio_metrics(w_sharpe, mu, S, RISK_FREE_RATE)
ret_mv, vol_mv, sr_mv = portfolio_metrics(w_minvol, mu, S, RISK_FREE_RATE)

plt.scatter(vol_sh, ret_sh, c='g', marker='*', s=200, label='Max Sharpe')
plt.scatter(vol_mv, ret_mv, c='r', marker='*', s=200, label='Min Volatility')

plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Efficient Frontier (TSLA forecast + historical BND & SPY)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "efficient_frontier.png"))
plt.show()

# ---------- Save results ----------
summary = {
    "used_forecast_file": used_forecast_file,
    "expected_returns": mu.to_dict(),
    "weights_max_sharpe": w_sharpe,
    "performance_max_sharpe": {"ret": ret_sh, "vol": vol_sh, "sharpe": sr_sh},
    "weights_min_vol": w_minvol,
    "performance_min_vol": {"ret": ret_mv, "vol": vol_mv, "sharpe": sr_mv}
}

with open(os.path.join(OUTPUT_DIR, "task4_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved summary to", os.path.join(OUTPUT_DIR, "task4_summary.json"))
print("Done.")
