"""
Task 2 - Time Series Forecasting (ARIMA + LSTM) for TSLA
Run: python task2_forecasting.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

# modeling
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# import preprocessing
from data_preprocessing import preprocess_data

sns.set(style="whitegrid")
np.random.seed(42)
tf.random.set_seed(42)

# ---------- Utility metrics ----------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-8, y_true))) * 100

# ---------- Load / Preprocess ----------
data = preprocess_data()  # uses default tickers + dates
pivot_close = data["pivot_close"]
# Prefer Adjusted Close if available. Here we have Close as pivot_close; adjust if using Adj Close.
series = pivot_close['TSLA'].copy()
series = series.dropna().sort_index()

# Train / Test split
train_end = "2023-12-31"
test_start = "2024-01-01"
train_series = series[:train_end]
test_series = series[test_start:]

print(f"Train observations: {len(train_series)}, Test observations: {len(test_series)}")

# ---------- Model 1: ARIMA on log-prices ----------
print("\n=== ARIMA (auto_arima) ===")
train_log = np.log(train_series)
# auto_arima finds p,d,q
arima_model = auto_arima(train_log, seasonal=False, stepwise=True, suppress_warnings=True, trace=True,
                         error_action='ignore', max_p=5, max_q=5, max_d=3)
print(arima_model.summary())

n_periods = len(test_series)
arima_forecast_log, conf_int = arima_model.predict(n_periods=n_periods, return_conf_int=True)
arima_forecast = np.exp(arima_forecast_log)  # invert log

# ---------- Evaluate ARIMA ----------
arima_mae = mean_absolute_error(test_series, arima_forecast)
arima_rmse = rmse(test_series, arima_forecast)
arima_mape = mape(test_series, arima_forecast)

print(f"ARIMA MAE: {arima_mae:.4f}, RMSE: {arima_rmse:.4f}, MAPE: {arima_mape:.2f}%")

# ---------- Model 2: LSTM ----------
print("\n=== LSTM ===")
# Prepare data
lookback = 60  # days
scaler = MinMaxScaler(feature_range=(0, 1))
ts_values = series.values.reshape(-1, 1)
ts_scaled = scaler.fit_transform(ts_values)

# Create sequences
def create_sequences(data_scaled, lookback):
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i - lookback:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(ts_scaled, lookback)

# Align indices: first target corresponds to index lookback
dates_all = series.index[lookback:]

# split by date
train_mask = dates_all <= train_end
test_mask = dates_all >= test_start

X_train = X_all[train_mask]
y_train = y_all[train_mask]

X_test = X_all[test_mask]
y_test = y_all[test_mask]

# reshape for LSTM [samples, timesteps, features]
X_train_resh = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_resh = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"LSTM train samples: {X_train_resh.shape[0]}, test samples: {X_test_resh.shape[0]}")

# Build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train_resh.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
history = model.fit(X_train_resh, y_train, validation_split=0.1, epochs=100, batch_size=32, callbacks=[es], verbose=1)

# Predict on test set
y_pred_scaled = model.predict(X_test_resh)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
y_test_true = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()

# Evaluate LSTM
lstm_mae = mean_absolute_error(y_test_true, y_pred)
lstm_rmse = rmse(y_test_true, y_pred)
lstm_mape = mape(y_test_true, y_pred)

print(f"LSTM MAE: {lstm_mae:.4f}, RMSE: {lstm_rmse:.4f}, MAPE: {lstm_mape:.2f}%")

# ---------- Plot results ----------
plt.figure(figsize=(14,6))
# actual test series
plt.plot(test_series.index, test_series.values, label='Actual', color='black')
# ARIMA forecast has same length as test but starting from test_start
plt.plot(test_series.index, arima_forecast, label='ARIMA Forecast', color='red')
# LSTM predictions correspond to dates_all[test_mask] (shifted because of lookback)
lstm_dates = dates_all[test_mask]
plt.plot(lstm_dates, y_pred, label='LSTM Forecast', color='green')
plt.title("TSLA: Actual vs Forecasts (ARIMA & LSTM)")
plt.legend()
plt.show()

# ---------- Save metrics ----------
metrics = {
    "arima": {"mae": arima_mae, "rmse": arima_rmse, "mape": arima_mape},
    "lstm": {"mae": lstm_mae, "rmse": lstm_rmse, "mape": lstm_mape}
}
with open("forecast_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved forecast_metrics.json")
