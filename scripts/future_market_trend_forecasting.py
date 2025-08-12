# task3_forecast.py
"""
Task 3 - Forecast Future Market Trends (ARIMA + LSTM)
Produces 6-month and 12-month forecasts with intervals for TSLA.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import math

# modeling libs
from pmdarima import auto_arima
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler

# local preprocessing function
from task1_data_preprocessing import preprocess_data

# ---------- Config ----------
LOOKBACK = 60                 # LSTM lookback used in Task 2
N_MC = 200                    # MC dropout forward passes
TRADING_DAYS_PER_MONTH = 21
HORIZON_6M = TRADING_DAYS_PER_MONTH * 6
HORIZON_12M = TRADING_DAYS_PER_MONTH * 12

ARIMA_PATH = "arima_model.pkl"
LSTM_PATH = "lstm_model.h5"
SCALER_PATH = "scaler.pkl"

os.makedirs("task3_outputs", exist_ok=True)

# ---------- Utilities ----------
def rmse(a, b):
    return math.sqrt(((a - b) ** 2).mean())

def mape(a, b):
    a, b = np.array(a), np.array(b)
    return np.mean(np.abs((a - b) / np.where(a == 0, 1e-8, a))) * 100

# ---------- Load data ----------
data = preprocess_data()
pivot_close = data["pivot_close"]
series = pivot_close['TSLA'].dropna().sort_index()

# forecast horizons
horizons = {"6m": HORIZON_6M, "12m": HORIZON_12M}

# ---------- ARIMA: load or train ----------
if os.path.exists(ARIMA_PATH):
    print("Loading ARIMA model from", ARIMA_PATH)
    arima_model = joblib.load(ARIMA_PATH)
else:
    print("Training ARIMA model (this may take a moment)...")
    train_end = "2023-12-31"
    train_series = series[:train_end]
    arima_model = auto_arima(np.log(train_series), seasonal=False, stepwise=True,
                             suppress_warnings=True, error_action='ignore', max_p=5, max_q=5)
    joblib.dump(arima_model, ARIMA_PATH)
    print("Saved ARIMA model to", ARIMA_PATH)

# ---------- LSTM: load scaler and model (or signal to retrain) ----------
scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

if os.path.exists(LSTM_PATH) and scaler is not None:
    print("Loading LSTM model and scaler")
    lstm_model = load_model(LSTM_PATH, compile=False)
else:
    print("LSTM model or scaler not found. Please run Task 2 LSTM training first to create", LSTM_PATH, "and", SCALER_PATH)
    lstm_model = None

# ---------- Forecast utilities for LSTM ----------
def create_sequences_from_series(values, lookback):
    X = []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i, 0])
    return np.array(X)

def mc_dropout_predict(model, X_input, n_mc=N_MC, batch_size=64):
    """
    Perform Monte-Carlo dropout: run model.predict with K.learning_phase set to 1 (dropout active).
    Returns array shape (n_mc, n_samples).
    """
    # Build function that runs model with learning_phase=1 to enable dropout at inference
    f = K.function([model.input, K.learning_phase()], [model.output])
    preds = []
    for _ in range(n_mc):
        p = f([X_input, 1])[0].ravel()
        preds.append(p)
    preds = np.array(preds)
    return preds

# ---------- Forecasting loop ----------
results_summary = {}
for label, n_steps in horizons.items():
    print(f"\nForecast horizon: {label} ({n_steps} trading days)")

    # -------- ARIMA multi-step forecast ----------
    # We will re-fit ARIMA on all available data (or you may prefer to use train-only)
    arima_refit = arima_model  # using the loaded model (trained on train data)
    # Forecast from last available date forward
    fc_log, conf_int = arima_refit.predict(n_periods=n_steps, return_conf_int=True)
    fc = np.exp(fc_log)
    arima_lower = np.exp(conf_int[:, 0])
    arima_upper = np.exp(conf_int[:, 1])

    # Dates for forecast: create trading-day index after last date
    last_date = series.index[-1]
    # create a simple sequence of business days using pandas date_range with B frequency
    fc_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps, freq='B')

    df_arima = pd.DataFrame({
        "date": fc_dates,
        "arima_forecast": fc,
        "arima_lower": arima_lower,
        "arima_upper": arima_upper
    }).set_index("date")

    # Save arima forecast
    arima_out_path = f"task3_outputs/forecast_arima_{label}.csv"
    df_arima.to_csv(arima_out_path)
    print("Saved ARIMA forecast to", arima_out_path)

    # -------- LSTM iterative multi-step forecasting using recursive strategy ----------
    df_lstm = pd.DataFrame(index=fc_dates)
    if lstm_model is not None and scaler is not None:
        # Prepare scaled series and seed sequence
        vals = series.values.reshape(-1, 1)
        scaled_vals = scaler.transform(vals)
        # last lookback window to initialize forecasting
        seed = scaled_vals[-LOOKBACK:].copy()
        preds_scaled = []
        # recursive one-step predictions for n_steps
        for step in range(n_steps):
            X_in = seed.reshape((1, LOOKBACK, 1))
            # MC dropout: get many predictions, take mean & quantiles
            if N_MC > 1:
                mc_preds = mc_dropout_predict(lstm_model, X_in, n_mc=N_MC)
                # mc_preds shape (n_mc, n_samples==1) -> flatten
                mc_preds = mc_preds.reshape(N_MC, -1)[:, 0]
                mean_pred = mc_preds.mean()
                lower = np.percentile(mc_preds, 2.5)
                upper = np.percentile(mc_preds, 97.5)
                # append mean and update seed with mean
                preds_scaled.append((mean_pred, lower, upper))
                # update seed by appending mean_pred and dropping first element
                seed = np.vstack([seed[1:], [[mean_pred]]])
            else:
                p = lstm_model.predict(X_in)[0, 0]
                preds_scaled.append((p, p, p))
                seed = np.vstack([seed[1:], [[p]]])

        preds_scaled = np.array(preds_scaled)  # shape (n_steps, 3)
        # invert scale to price space
        mean_scaled = preds_scaled[:, 0].reshape(-1, 1)
        lower_scaled = preds_scaled[:, 1].reshape(-1, 1)
        upper_scaled = preds_scaled[:, 2].reshape(-1, 1)

        mean_price = scaler.inverse_transform(mean_scaled).ravel()
        lower_price = scaler.inverse_transform(lower_scaled).ravel()
        upper_price = scaler.inverse_transform(upper_scaled).ravel()

        df_lstm["lstm_forecast"] = mean_price
        df_lstm["lstm_lower"] = lower_price
        df_lstm["lstm_upper"] = upper_price

        # Save LSTM forecast
        lstm_out_path = f"task3_outputs/forecast_lstm_{label}.csv"
        df_lstm.to_csv(lstm_out_path)
        print("Saved LSTM forecast to", lstm_out_path)
    else:
        print("LSTM model/scaler not available; skipping LSTM forecast for", label)

    # ---------- Combine for summary (if both exist) ----------
    combined = df_arima.join(df_lstm, how='outer')
    combined.to_csv(f"task3_outputs/forecast_combined_{label}.csv")

    # add to results summary
    results_summary[label] = {
        "arima_csv": arima_out_path,
        "lstm_csv": lstm_out_path if lstm_model is not None else None,
        "combined_csv": f"task3_outputs/forecast_combined_{label}.csv"
    }

# ---------- Plot combined charts ----------
plt.figure(figsize=(14, 8))
plt.plot(series.index, series.values, label="Historical (TSLA)", color="black")

# overlay 6m ARIMA and LSTM
for label, style in [("6m", "--"), ("12m", ":")]:
    arima_df = pd.read_csv(f"task3_outputs/forecast_arima_{label}.csv", parse_dates=["date"], index_col="date")
    plt.plot(arima_df.index, arima_df["arima_forecast"], label=f"ARIMA {label}", linestyle=style)
    plt.fill_between(arima_df.index, arima_df["arima_lower"], arima_df["arima_upper"], alpha=0.15)

    lstm_path = f"task3_outputs/forecast_lstm_{label}.csv"
    if os.path.exists(lstm_path):
        lstm_df = pd.read_csv(lstm_path, parse_dates=["date"], index_col="date")
        plt.plot(lstm_df.index, lstm_df["lstm_forecast"], label=f"LSTM {label}", linestyle=style, color="green")
        plt.fill_between(lstm_df.index, lstm_df["lstm_lower"], lstm_df["lstm_upper"], alpha=0.12, color="green")

plt.title("TSLA Historical + ARIMA & LSTM Forecasts (6m & 12m)")
plt.legend()
plt.tight_layout()
plt.savefig("task3_outputs/task3_forecast_plots.png")
print("Saved plot to task3_outputs/task3_forecast_plots.png")
plt.show()

# ---------- Save summary ----------
with open("task3_outputs/task3_forecast_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

print("\nDone. Outputs saved in ./task3_outputs/")
