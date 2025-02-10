import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from scipy.stats.mstats import winsorize
from threading import Thread

def get_stock_data(ticker, start_date, end_date, interval):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if data.empty:
        raise ValueError("No data returned.")
    if 'Adj Close' in data.columns:
        price_column = 'Adj Close'
        data['Returns'] = data[price_column].pct_change()
    elif 'Close' in data.columns:
        price_column = 'Close'
        data['Returns'] = data[price_column].pct_change()
    else:
        raise KeyError("No price column available.")
    volume_array = data['Volume'].values
    volume_winsorized = winsorize(volume_array, limits=[0.01, 0.01])
    data['Volume'] = pd.Series(volume_winsorized.data.ravel(), index=data.index)
    price_series = data[price_column].squeeze()
    data['Log_Returns'] = np.log(price_series / price_series.shift(1))
    data['Rolling_Close'] = price_series.rolling(window=10).mean()
    data['SMA_20'] = SMAIndicator(price_series, window=20).sma_indicator()
    data['SMA_50'] = SMAIndicator(price_series, window=50).sma_indicator()
    macd_indicator = MACD(price_series, window_slow=26, window_fast=12, window_sign=9)
    data['MACD'] = macd_indicator.macd()
    data['MACD_signal'] = macd_indicator.macd_signal()
    data['RSI'] = RSIIndicator(price_series, window=14).rsi()
    data.dropna(inplace=True)
    return data, price_column

def create_sequences(data, look_back=50):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:i+look_back, :])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)

def run_prediction(ticker, start_date, end_date, interval):
    loading_label.config(text="Loading, please wait...")
    start_button.config(state="disabled")
    try:
        data, price_column = get_stock_data(ticker, start_date, end_date, interval)
        data['Target'] = data[price_column].shift(-1)
        data.dropna(inplace=True)
        X_rf = data[[price_column, 'Volume', 'Returns', 'SMA_20', 'SMA_50', 'RSI', 'Log_Returns', 'Rolling_Close', 'MACD', 'MACD_signal']]
        y_rf = data['Target']
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
        rf = RandomForestRegressor(random_state=42)
        rf_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
        rf_search.fit(X_train_rf, y_train_rf)
        best_rf = rf_search.best_estimator_
        rf_predictions = best_rf.predict(X_test_rf)
        feature_data = data[[price_column, 'Volume', 'Returns', 'SMA_20', 'SMA_50', 'RSI', 'Log_Returns', 'Rolling_Close', 'MACD', 'MACD_signal']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)
        look_back = 50
        X_lstm, y_lstm = create_sequences(scaled_data, look_back)
        split = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
        y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]
        model = Sequential([
            Bidirectional(LSTM(50, return_sequences=True, input_shape=(look_back, scaled_data.shape[1]))),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_split=0.1)
        lstm_predictions = model.predict(X_test_lstm)
        lstm_full = np.concatenate([lstm_predictions, np.zeros((lstm_predictions.shape[0], scaled_data.shape[1]-1))], axis=1)
        lstm_rescaled = scaler.inverse_transform(lstm_full)[:, 0]
        y_full = np.concatenate([y_test_lstm.reshape(-1, 1), np.zeros((y_test_lstm.shape[0], scaled_data.shape[1]-1))], axis=1)
        y_actual = scaler.inverse_transform(y_full)[:, 0]
        length = len(y_actual)
        ensemble = (rf_predictions[-length:] + lstm_rescaled) / 2
        final_preds = pd.Series(ensemble).rolling(window=5).mean().fillna(method='bfill').values
        evaluate_model(y_actual, final_preds)
        dates = data.index[-length:]
        plt.figure(figsize=(10,6))
        plt.plot(dates, y_actual, label="Actual")
        plt.plot(dates, final_preds, label="Prediction")
        plt.legend()
        plt.show()
    except Exception as e:
        print(str(e))
    loading_label.config(text="")
    start_button.config(state="normal")

def on_start():
    t = Thread(target=run_prediction, args=(ticker_entry.get(), start_entry.get(), end_entry.get(), interval_map[interval_combo.get()]))
    t.start()

root = tk.Tk()
root.title("Stock Predictor")
ticker_label = tk.Label(root, text="Ticker:")
ticker_label.pack()
ticker_entry = tk.Entry(root)
ticker_entry.pack()
start_label = tk.Label(root, text="Start Date (YYYY-MM-DD):")
start_label.pack()
start_entry = tk.Entry(root)
start_entry.pack()
end_label = tk.Label(root, text="End Date (YYYY-MM-DD):")
end_label.pack()
end_entry = tk.Entry(root)
end_entry.pack()
interval_label = tk.Label(root, text="Interval:")
interval_label.pack()
interval_combo = ttk.Combobox(root, values=["Daily","Weekly","Monthly"])
interval_combo.current(0)
interval_combo.pack()
interval_map = {"Daily":"1d", "Weekly":"1wk", "Monthly":"1mo"}
start_button = tk.Button(root, text="Start", command=on_start)
start_button.pack()
loading_label = tk.Label(root, text="")
loading_label.pack()
root.mainloop()
