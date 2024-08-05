from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBClassifier

def build_arima_model(data, order=(1,1,1)):
    model = ARIMA(data, order=order)
    return model.fit()

def build_arimax_model(data, exog, order=(1,1,1)):
    model = ARIMA(data, exog=exog, order=order)
    return model.fit()

def build_xgboost_model(features, target, max_depth=3, n_estimators=100, random_state=42):
    model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=random_state)
    model.fit(features, target)
    return model
