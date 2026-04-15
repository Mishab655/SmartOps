import pandas as pd
df = pd.read_csv('ML_datasets/forecast_df.csv')
def prepare_time_series(forecast_df, category=None, freq="D"):
    """
    Prepare time series for ARIMA and Prophet.
    """

    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if category:
        df = df[df["product_category"] == category]

    # Aggregate in case of duplicates
    df = df.groupby("date")["total_sales"].sum().reset_index()

    # Set index
    df = df.set_index("date").asfreq(freq)

    # Fill missing values (important)
    df["total_sales"] = df["total_sales"].fillna(0)

    return df

def train_test_split_ts(df, split_date):
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    return train, test


df_ts = prepare_time_series(df, category="bed_bath_table")
train, test = train_test_split_ts(df_ts, "2018-06-01")


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_arima(train, test, order=(1,1,1)):
    model = ARIMA(train["total_sales"], order=order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    mae = mean_absolute_error(test["total_sales"], forecast)
    rmse = np.sqrt(mean_squared_error(test["total_sales"], forecast))

    return model_fit, forecast, {"MAE": mae, "RMSE": rmse}

from prophet import Prophet

def train_prophet(train, test):

    train_prophet = train.reset_index().rename(
        columns={"date": "ds", "total_sales": "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_prophet)

    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast = model.predict(future)

    forecast_test = forecast.set_index("ds").loc[test.index]["yhat"]

    mae = mean_absolute_error(test["total_sales"], forecast_test)
    rmse = np.sqrt(mean_squared_error(test["total_sales"], forecast_test))

    return model, forecast_test, {"MAE": mae, "RMSE": rmse}


arima_model, arima_pred, arima_metrics = train_arima(train, test)
prophet_model, prophet_pred, prophet_metrics = train_prophet(train, test)

print("ARIMA:", arima_metrics)
print("Prophet:", prophet_metrics)


import matplotlib.pyplot as plt

def plot_forecasts(train, test, arima_pred, prophet_pred):

    plt.figure(figsize=(12,6))
    plt.plot(train.index, train["total_sales"], label="Train")
    plt.plot(test.index, test["total_sales"], label="Actual")
    plt.plot(test.index, arima_pred, label="ARIMA")
    plt.plot(test.index, prophet_pred, label="Prophet")

    plt.legend()
    plt.title("ARIMA vs Prophet Forecast")
    plt.show()