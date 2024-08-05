import pandas as pd
from statsmodels.tsa.stattools import adfuller

def convert_to_datetime(df, date_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    return df

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    return result[1]  # p-value

def difference_data(df, column_name):
    df[column_name] = df[column_name].diff().dropna()
    return df
