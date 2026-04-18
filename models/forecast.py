import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.db import get_db_engine
from datetime import datetime


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


def train_arima(train, test, order=(1, 1, 1)):
    model = ARIMA(train["total_sales"], order=order)
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=len(test))

    mae = mean_absolute_error(test["total_sales"], forecast)
    rmse = np.sqrt(mean_squared_error(test["total_sales"], forecast))

    return model_fit, forecast, {"MAE": mae, "RMSE": rmse}


def train_prophet(train, test):
    train_prophet_df = train.reset_index().rename(
        columns={"date": "ds", "total_sales": "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_prophet_df)

    future = model.make_future_dataframe(periods=len(test), freq="D")
    forecast = model.predict(future)

    forecast_test = forecast.set_index("ds").loc[test.index]["yhat"]

    mae = mean_absolute_error(test["total_sales"], forecast_test)
    rmse = np.sqrt(mean_squared_error(test["total_sales"], forecast_test))

    return model, forecast_test, {"MAE": mae, "RMSE": rmse}


def plot_forecasts(train, test, arima_pred, prophet_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train["total_sales"], label="Train")
    plt.plot(test.index, test["total_sales"], label="Actual")
    plt.plot(test.index, arima_pred, label="ARIMA")
    plt.plot(test.index, prophet_pred, label="Prophet")

    plt.legend()
    plt.title("ARIMA vs Prophet Forecast")
    plt.show()


def save_forecasts_to_db(forecast_pred, category_name, model_version="arima_v1", actual_sales=None):
    """
    Saves the generated forecasts to the category_forecast table in postgres.
    actual_sales: optional pd.Series aligned with forecast_pred index.
                  Pass test['total_sales'] for the test period; leave None for future dates.
    """
    engine = get_db_engine()

    # Format forecasts for DB
    predictions = forecast_pred.reset_index()
    predictions.columns = ['forecast_date', 'predicted_sales']

    # Add metadata columns
    predictions['product_category'] = category_name
    predictions['model_version'] = model_version
    predictions['created_at'] = datetime.now()

    # FIX A: Populate actual_sales when we have them (test period)
    if actual_sales is not None:
        predictions['actual_sales'] = actual_sales.values
    else:
        predictions['actual_sales'] = None  # Genuinely unknown for future dates

    # Rearrange columns to match table schema
    cols = ['forecast_date', 'product_category', 'actual_sales', 'predicted_sales', 'model_version', 'created_at']
    predictions = predictions[cols]

    try:
        predictions.to_sql("category_forecast", engine, if_exists="append", index=False)
        print(f"Successfully saved {len(predictions)} forecast records to DB for {category_name}.")
    except Exception as e:
        print(f"Error saving to DB: {e}")


def run_forecast_for_all_categories(forecast_df, split_date="2018-06-01", future_days=90, model_version="arima_v1"):
    """
    Trains ARIMA for every product category in the dataset and saves all forecasts to DB.
    - Saves test-period predictions WITH actual_sales populated (Fix A).
    - Retrains on the full dataset and appends `future_days` of genuine future forecasts (Fix B).
    """
    categories = forecast_df["product_category"].unique()
    print(f"Found {len(categories)} categories. Training ARIMA for each...")

    for category in categories:
        try:
            df_ts = prepare_time_series(forecast_df, category=category)

            # Skip categories with insufficient data
            if len(df_ts) < 30:
                print(f"  Skipping '{category}': not enough data ({len(df_ts)} rows).")
                continue

            train, test = train_test_split_ts(df_ts, split_date)

            if len(train) < 10 or len(test) < 1:
                print(f"  Skipping '{category}': train/test split too small.")
                continue

            # Step 1: Save test-period predictions WITH actual_sales (Fix A)
            _, arima_pred, arima_metrics = train_arima(train, test)
            print(f"  [{category}] ARIMA test metrics: {arima_metrics}")
            save_forecasts_to_db(
                arima_pred,
                category_name=category,
                model_version=model_version,
                actual_sales=test["total_sales"],  # FIX A: pass real actuals
            )

            # Step 2: Retrain on full data and forecast future dates (Fix B)
            full_model = ARIMA(df_ts["total_sales"], order=(1, 1, 1)).fit()
            last_date = df_ts.index[-1]
            future_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=future_days,
                freq="D",
            )
            future_pred = pd.Series(
                full_model.forecast(steps=future_days).values,
                index=future_index,
                name="total_sales",
            )
            future_pred.index.name = "date"
            print(f"  [{category}] Saving {future_days}-day future forecast from {future_index[0].date()} to {future_index[-1].date()}.")
            save_forecasts_to_db(
                future_pred,
                category_name=category,
                model_version=model_version,
                actual_sales=None,  # Genuinely unknown for future
            )

        except Exception as e:
            print(f"  Error processing '{category}': {e}")

    print("All categories processed.")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ML_datasets/forecast_df.csv')
    df = pd.read_csv(data_path)

    # --- Run for ALL categories (Issue 3 fix: no longer hardcoded to one category) ---
    run_forecast_for_all_categories(df, split_date="2018-06-01", model_version="arima_v1")

    # --- Optional: plot for a single category to visually verify ---
    df_ts = prepare_time_series(df, category="bed_bath_table")
    train, test = train_test_split_ts(df_ts, "2018-06-01")
    arima_model, arima_pred, arima_metrics = train_arima(train, test)
    prophet_model, prophet_pred, prophet_metrics = train_prophet(train, test)

    print("ARIMA:", arima_metrics)
    print("Prophet:", prophet_metrics)

    plot_forecasts(train, test, arima_pred, prophet_pred)