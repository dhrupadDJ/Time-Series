from data_preprocessing.load_data import load_data
from data_preprocessing.preprocess_data import convert_to_datetime, difference_data, check_stationarity
from models.build_model import build_arima_model, build_arimax_model, build_xgboost_model
from evaluation.evaluate_model import evaluate_arima, evaluate_xgboost
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score

# Load and preprocess data
file_path = r"C:\Users\jaisw\Desktop\Data Science Final Project\Apple_Stock_Price_Forecasting\Apple_Stock_Price_Forecasting\data\AAPL.csv"
data = load_data(file_path)
data = convert_to_datetime(data, 'Date')

# Univariate analysis
data_diff = difference_data(data.copy(), 'AAPL')
p_value = check_stationarity(data_diff['AAPL'].dropna())
print(f'Stationarity p-value: {p_value}')

# ARIMA model
arima_model = build_arima_model(data['AAPL'])
print(arima_model.summary())

# Forecast with ARIMA
forecast = arima_model.get_forecast(steps=2)
ypred = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)
print(ypred)

# Plot ARIMA
plt.plot(data['AAPL'], label='Actual')
plt.plot(ypred, label='Forecast', color='orange')
plt.fill_between(ypred.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.15)
plt.legend()
plt.show()

# Bivariate analysis with ARIMAX
exog_data = data[['TXN']].iloc[:-2]
arimax_model = build_arimax_model(data['AAPL'], exog=exog_data)
print(arimax_model.summary())

# Forecast with ARIMAX
exog_forecast = data['TXN'].iloc[-2:]
forecast_arimax = arimax_model.get_forecast(steps=2, exog=exog_forecast)
ypred_arimax = forecast_arimax.predicted_mean
conf_int_arimax = forecast_arimax.conf_int(alpha=0.05)

# Plot ARIMAX
plt.plot(data['AAPL'], label='Actual')
plt.plot(ypred_arimax, label='Forecast with Exog', color='orange')
plt.fill_between(ypred_arimax.index, conf_int_arimax.iloc[:, 0], conf_int_arimax.iloc[:, 1], color='k', alpha=0.15)
plt.legend()
plt.show()

# XGBoost Model
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
train = data.iloc[:-30]
test = data.iloc[-30:]

xgb_model = build_xgboost_model(train[features], train['Target'])
xgb_preds = xgb_model.predict(test[features])

# Evaluate XGBoost Model
precision = evaluate_xgboost(test['Target'], xgb_preds)
print(f'XGBoost Precision: {precision}')

# Backtesting
def backtest(data, model, features, start=5031, step=120):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:(i+step)].copy()
        model.fit(train[features], train['Target'])
        preds = model.predict(test[features])
        preds = pd.Series(preds, index=test.index, name='predictions')
        combine = pd.concat([test['Target'], preds], axis=1)
        all_predictions.append(combine)
    return pd.concat(all_predictions)

predictions = backtest(data, xgb_model, features)
precision_backtest = precision_score(predictions['Target'], predictions['predictions'])
print(f'Backtest Precision: {precision_backtest}')
