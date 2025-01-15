import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data returned. Check the ticker or date range.")

    if 'Adj Close' in data.columns:
        data['Returns'] = data['Adj Close'].pct_change()
        price_column = 'Adj Close'
    elif 'Close' in data.columns:
        data['Returns'] = data['Close'].pct_change()
        price_column = 'Close'
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' columns are available.")

    # Ensure we pass a 1D Series to SMAIndicator and RSIIndicator
    data['SMA_20'] = SMAIndicator(data[price_column].squeeze(), window=20).sma_indicator()
    data['RSI'] = RSIIndicator(data[price_column].squeeze(), window=14).rsi()

    data.dropna(inplace=True)
    return data, price_column

def create_sequences(data, look_back=50):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:i+look_back, 0])
        y.append(data[i+look_back, 0])
    return np.array(X), np.array(y)

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

def plot_predictions(test_dates, y_test_smoothed, predictions_smoothed, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_smoothed, label="Actual Prices", alpha=0.8, linewidth=2)
    plt.plot(test_dates, predictions_smoothed, label=f"Predicted Prices ({model_name})", linestyle='--', alpha=0.8, linewidth=2)
    plt.title(f"{model_name} Predictions")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

ticker = "AAPL"
start_date = "2015-01-01"
end_date = "2023-01-01"

stock_data, price_column = get_stock_data(ticker, start_date, end_date)
if len(stock_data) < 2:
    raise ValueError("Not enough data points after dropping NaN rows.")

stock_data['Target'] = stock_data[price_column].shift(-1)
stock_data.dropna(inplace=True)
if len(stock_data) < 2:
    raise ValueError("Not enough data left to train or test after shift/dropna.")

X_rf = stock_data[[price_column, 'Volume', 'Returns', 'SMA_20', 'RSI']]
y_rf = stock_data['Target']

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
rf_predictions = rf_model.predict(X_test_rf)

evaluate_model(y_test_rf, rf_predictions, "Random Forest")

y_test_rf_smoothed = pd.Series(y_test_rf.values).rolling(window=10).mean()
rf_predictions_smoothed = pd.Series(rf_predictions).rolling(window=10).mean()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[[price_column]])
look_back = 50
if len(stock_data) <= look_back + 1:
    raise ValueError("Not enough data to create LSTM sequences based on the specified look_back.")

X_lstm, y_lstm = create_sequences(scaled_data, look_back)

split = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_split=0.1)

lstm_predictions = model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

evaluate_model(y_test_actual, lstm_predictions, "LSTM")

y_test_actual_smoothed = pd.Series(y_test_actual.flatten()).rolling(window=10).mean()
lstm_predictions_smoothed = pd.Series(lstm_predictions.flatten()).rolling(window=10).mean()

test_dates_rf = stock_data.index[-len(y_test_rf):]
test_dates_lstm = stock_data.index[-len(y_test_lstm):]

plot_predictions(test_dates_rf, y_test_rf_smoothed, rf_predictions_smoothed, "Random Forest")
plot_predictions(test_dates_lstm, y_test_actual_smoothed, lstm_predictions_smoothed, "LSTM")
