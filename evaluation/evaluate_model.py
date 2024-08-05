import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score

def evaluate_arima(predictions, actual):
    mae = mean_absolute_error(actual, predictions)
    return mae

def evaluate_xgboost(test, predictions):
    precision = precision_score(test, predictions)
    return precision
