import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from scipy.stats.mstats import winsorize
import plotly.graph_objects as go

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
    volume_array = data['Volume'].values
    volume_winsorized = winsorize(volume_array, limits=[0.01, 0.01])
    data['Volume'] = pd.Series(volume_winsorized.data.ravel(), index=data.index, name='Volume')
    data['Log_Returns'] = np.log(data[price_column] / data[price_column].shift(1))
    data['Rolling_Close'] = data[price_column].rolling(window=10).mean()
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
    print(model_name+" Performance:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 Score:", r2)

def plot_predictions_with_plotly(dates, actual, predicted, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Stock Price')
    fig.show()

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

X_rf = stock_data[[price_column, 'Volume', 'Returns', 'SMA_20', 'RSI', 'Log_Returns', 'Rolling_Close']]
y_rf = stock_data['Target']
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_model = RandomForestRegressor(random_state=42)
rf_search = RandomizedSearchCV(rf_model, param_distributions=param_grid, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
rf_search.fit(X_train_rf, y_train_rf)
best_rf = rf_search.best_estimator_
rf_predictions = best_rf.predict(X_test_rf)
evaluate_model(y_test_rf, rf_predictions, "Random Forest")

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
    Bidirectional(LSTM(50, return_sequences=True, input_shape=(look_back, 1))),
    Dropout(0.2),
    Bidirectional(LSTM(50, return_sequences=False)),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_split=0.1)

lstm_predictions = model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
evaluate_model(y_test_actual, lstm_predictions, "Bidirectional LSTM")

combined_predictions = (rf_predictions[-len(lstm_predictions):] + lstm_predictions.flatten()) / 2
evaluate_model(y_test_actual, combined_predictions, "Ensemble")

test_dates_rf = stock_data.index[-len(y_test_rf):]
test_dates_lstm = stock_data.index[-len(y_test_lstm):]

plot_predictions_with_plotly(test_dates_rf, y_test_rf.values, rf_predictions, "Random Forest Predictions")
plot_predictions_with_plotly(test_dates_lstm, y_test_actual.flatten(), lstm_predictions.flatten(), "Bidirectional LSTM Predictions")
plot_predictions_with_plotly(test_dates_lstm, y_test_actual.flatten(), combined_predictions, "Ensemble Predictions")
