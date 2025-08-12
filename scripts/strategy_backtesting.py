# task5_backtest.py
"""
Task 5 - Backtest strategy vs 60/40 benchmark
Outputs: cumulative returns plot, metrics JSON, and CSV of portfolio daily NAVs.

run:  python task5_backtest.py --mode monthly_rebalance

"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use("seaborn")

# Local imports (preprocessing)
from task1_data_preprocessing import preprocess_data

OUTPUT_DIR = "task5_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Config ----------
BACKTEST_START = pd.to_datetime("2024-08-01")
BACKTEST_END   = pd.to_datetime("2025-07-31")
INITIAL_CAPITAL = 100_000
RISK_FREE = 0.03
BENCHMARK_WEIGHTS = {"SPY": 0.6, "BND": 0.4}
WEIGHT_SOURCE_JSON = "task4_outputs/task4_summary.json"
REBALANCE_MODE = "monthly_rebalance"  # or "hold"
TRANSACTION_COST = 0.0  # proportion per trade, e.g., 0.001 for 0.1%

# ---------- Utilities ----------
def annualized_return(cum_return, days):
    return (1 + cum_return) ** (252.0/days) - 1

def annualized_vol(returns):
    return returns.std() * np.sqrt(252)

def sharpe_ratio(returns, rf=RISK_FREE):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    return (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan

# ---------- Load data ----------
data = preprocess_data()
prices = data["pivot_close"][['TSLA','BND','SPY']].sort_index()
prices = prices.loc[BACKTEST_START - pd.Timedelta(days=60): BACKTEST_END]  # keep enough history if lookback needed
prices = prices.dropna(how='all')  # drop days with no prices at all

# ---------- Get strategy weights ----------
if os.path.exists(WEIGHT_SOURCE_JSON):
    with open(WEIGHT_SOURCE_JSON, "r") as f:
        summary = json.load(f)
    # prefer max_sharpe weights
    if "weights_max_sharpe" in summary:
        strat_weights = summary["weights_max_sharpe"]
    elif "weights" in summary:
        strat_weights = summary["weights"]
    else:
        raise KeyError("No suitable weights in JSON")
    # ensure all assets keys exist in our price set; fill missing with 0
    for asset in ["TSLA","BND","SPY"]:
        if asset not in strat_weights:
            strat_weights[asset] = 0.0
else:
    # fallback example: 30% TSLA, 40% SPY, 30% BND
    strat_weights = {"TSLA":0.3, "SPY":0.4, "BND":0.3}

print("Using strategy weights:", strat_weights)
print("Benchmark weights:", BENCHMARK_WEIGHTS)

# ---------- Prepare backtest price series ----------
prices_bt = prices.loc[BACKTEST_START:BACKTEST_END].copy()
if prices_bt.empty:
    raise ValueError("No price data in backtest window. Check data and dates.")

# daily returns
rets_bt = prices_bt.pct_change().dropna()

# ---------- Create date index for rebalancing ----------
if REBALANCE_MODE == "monthly_rebalance":
    # find first trading day of each month in backtest window
    months = pd.date_range(BACKTEST_START, BACKTEST_END, freq='MS')
    rebalance_dates = []
    for m in months:
        # get first index >= month start
        idx = prices_bt.index.searchsorted(m)
        if idx < len(prices_bt.index):
            rebalance_dates.append(prices_bt.index[idx])
else:
    rebalance_dates = [prices_bt.index[0]]  # initial only

# ---------- Backtest logic ----------
def run_backtest(weights, mode="monthly_rebalance", tx_cost=TRANSACTION_COST):
    cash = INITIAL_CAPITAL
    nav = []  # daily NAV
    holdings = {asset:0.0 for asset in ["TSLA","BND","SPY"]}
    current_weights = {k:0.0 for k in holdings.keys()}

    # Start by setting positions on first day (rebalance)
    dates = prices_bt.index
    first_day = dates[0]

    # set initial allocation at first_day
    def allocate_at_date(date, weights, cash_available, tx_cost):
        p = prices_bt.loc[date]
        # compute target dollar allocation
        target_dollars = {asset: weights.get(asset,0.0)*cash_available for asset in holdings.keys()}
        # compute target shares (floor to fractional allowed - allow fractional)
        target_shares = {asset: (target_dollars[asset] / p[asset]) if p[asset] > 0 else 0.0 for asset in holdings.keys()}
        return target_shares

    # initial allocation
    target_shares = allocate_at_date(first_day, weights, cash, tx_cost)
    holdings = target_shares.copy()
    # compute cash left (we allow fractional, so exact)
    cash = 0.0  # fully invested

    # record NAV on first day
    nav.append((first_day, sum(holdings[a]*prices_bt.loc[first_day,a] for a in holdings.keys()) + cash))

    # prepare set of rebalance dates
    if mode == "monthly_rebalance":
        rebalance_set = set(rebalance_dates)
    else:
        rebalance_set = set([first_day])

    # track previous holdings (for computing trades)
    prev_holdings = holdings.copy()

    for date in dates[1:]:
        # Rebalance if date in rebalance_set
        if date in rebalance_set:
            # compute portfolio value before rebalance
            port_val = sum(prev_holdings[a]*prices_bt.loc[date,a] for a in holdings.keys()) + cash
            # determine new target shares
            new_shares = allocate_at_date(date, weights, port_val, tx_cost)
            # optional: compute transaction costs if modeling (ignored now)
            # update holdings
            prev_holdings = new_shares.copy()
            holdings = new_shares.copy()
            cash = 0.0

        # compute NAV for the day
        today_nav = sum(prev_holdings[a]*prices_bt.loc[date,a] for a in holdings.keys()) + cash
        nav.append((date, today_nav))

    nav_df = pd.DataFrame(nav, columns=["date","nav"]).set_index("date")
    nav_df["returns"] = nav_df["nav"].pct_change().fillna(0.0)
    return nav_df

# run strategy backtest
strategy_nav = run_backtest(strat_weights, mode=REBALANCE_MODE)
# run benchmark backtest: only SPY & BND
bench_weights_full = {"TSLA":0.0, "SPY":BENCHMARK_WEIGHTS["SPY"], "BND":BENCHMARK_WEIGHTS["BND"]}
benchmark_nav = run_backtest(bench_weights_full, mode=REBALANCE_MODE)

# ---------- Metrics ----------
def compute_metrics(nav_df):
    total_ret = nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1.0
    days = (nav_df.index[-1] - nav_df.index[0]).days
    ann_ret = annualized_return(total_ret, days)
    ann_vol = annualized_vol(nav_df["returns"])
    sr = (nav_df["returns"].mean()*252 - RISK_FREE) / (nav_df["returns"].std()*np.sqrt(252)) if nav_df["returns"].std() > 0 else np.nan
    return {"total_return": total_ret, "annualized_return": ann_ret, "annualized_vol": ann_vol, "sharpe": sr}

strat_metrics = compute_metrics(strategy_nav)
bench_metrics = compute_metrics(benchmark_nav)

# ---------- Output & Plot ----------
# Merge NAVs
merged = pd.DataFrame({
    "strategy_nav": strategy_nav["nav"],
    "benchmark_nav": benchmark_nav["nav"]
}).dropna()

merged["strategy_cumret"] = merged["strategy_nav"] / merged["strategy_nav"].iloc[0]
merged["benchmark_cumret"] = merged["benchmark_nav"] / merged["benchmark_nav"].iloc[0]

plt.figure(figsize=(12,6))
plt.plot(merged.index, merged["strategy_cumret"], label="Strategy (from Task 4)")
plt.plot(merged.index, merged["benchmark_cumret"], label="Benchmark 60/40 SPY/BND", linestyle="--")
plt.title("Backtest: Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (Normalized)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "backtest_cumulative.png"), dpi=300)
plt.show()

# save navs and metrics
strategy_nav.to_csv(os.path.join(OUTPUT_DIR, "strategy_nav.csv"))
benchmark_nav.to_csv(os.path.join(OUTPUT_DIR, "benchmark_nav.csv"))
with open(os.path.join(OUTPUT_DIR, "task5_metrics.json"), "w") as f:
    json.dump({"strategy": strat_metrics, "benchmark": bench_metrics, "weights": strat_weights}, f, indent=2)

print("Strategy metrics:", strat_metrics)
print("Benchmark metrics:", bench_metrics)
print("Outputs saved to", OUTPUT_DIR)
