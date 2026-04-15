import pandas as pd
import numpy as np

def build_category_sales_forecasting_dataset(
    orders,
    order_items,
    products,
    translation,
    order_status="delivered"
):

    orders = orders.copy()
    order_items = order_items.copy()
    products = products.copy()
    translation = translation.copy()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    orders = orders[orders["order_status"] == order_status]

    df = (
        orders
        .merge(order_items, on="order_id")
        .merge(products, on="product_id")
        .merge(translation, on="product_category_name", how="left")
    )

    df["date"] = df["order_purchase_timestamp"].dt.date

    forecast_df = (
        df.groupby(["date", "product_category_name_english"])
        .agg(total_sales=("price", "sum"))
        .reset_index()
    )

    forecast_df.columns = ["date", "product_category", "total_sales"]

    return forecast_df

import pandas as pd

def build_churn_dataset(
    orders,
    customers,
    order_items,
    cutoff_date="2018-05-01",
    min_frequency=2
):
    """
    Builds a customer-level RFM scoring dataset for churn risk assessment.

    Approach: Pure RFM Scoring (no binary churn label).

    Rationale: Olist data is structurally one-purchase-dominant (~95% of
    customers never repeat). Any binary churn label derived from a future
    window would be overwhelmingly imbalanced and produce a meaningless
    classifier. Instead, we compute interpretable RFM scores from the
    observation window and rank customers by churn risk (High/Medium/Low).

    Only repeat customers (frequency >= min_frequency) are included,
    as churning is only meaningful in the context of customers who have
    demonstrated return purchase intent.
    """

    orders = orders.copy()
    customers = customers.copy()
    order_items = order_items.copy()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"]
    )

    cutoff_date = pd.to_datetime(cutoff_date)

    # Use only observation window (before cutoff)
    observation = orders[orders["order_purchase_timestamp"] < cutoff_date]

    # Merge for feature building
    df = observation.merge(customers, on="customer_id", how="left")
    df = df.merge(order_items, on="order_id", how="left")

    reference_date = observation["order_purchase_timestamp"].max()

    rfm_df = (
        df.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp",
                     lambda x: (reference_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("price", "sum")
        )
        .reset_index()
    )

    # Focus on repeat customers only
    rfm_df = rfm_df[rfm_df["frequency"] >= min_frequency].copy()

    # --- RFM Quantile Scoring (3=Best, 1=Worst for all dimensions) ---
    # Recency: lower days = more recent = better → score 3 for low recency
    rfm_df["r_score"] = pd.qcut(rfm_df["recency"], q=3, labels=[3, 2, 1]).astype(int)

    # Frequency: higher = better → score 3 for high frequency
    rfm_df["f_score"] = pd.qcut(
        rfm_df["frequency"].rank(method="first"), q=3, labels=[1, 2, 3]
    ).astype(int)

    # Monetary: higher = better → score 3 for high spend
    rfm_df["m_score"] = pd.qcut(
        rfm_df["monetary"].rank(method="first"), q=3, labels=[1, 2, 3]
    ).astype(int)

    # Composite RFM Risk Score: low score = high churn risk
    rfm_df["rfm_risk_score"] = rfm_df["r_score"] + rfm_df["f_score"] + rfm_df["m_score"]

    # Churn risk band (High risk = low RFM score, Low risk = high RFM score)
    rfm_df["churn_risk"] = pd.cut(
        rfm_df["rfm_risk_score"],
        bins=[2, 5, 7, 9],
        labels=["High", "Medium", "Low"],
        include_lowest=True
    )

    return rfm_df



def build_category_sentiment_dataset(
    reviews: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds a dataset for sentiment analysis
    grouped by product category.
    """

    reviews = reviews.copy()
    orders = orders.copy()
    order_items = order_items.copy()
    products = products.copy()
    translation = translation.copy()


    # Join reviews with orders
    df = reviews.merge(orders, on="order_id", how="inner")

    # Join with order items
    df = df.merge(order_items, on="order_id", how="inner")

    # Join with products
    df = df.merge(products, on="product_id", how="left")
    
    df = df.merge(translation, on='product_category_name', how='left')

    # Select relevant columns
    sentiment_df = df[
        [
            "product_category_name_english",
            "review_score",
            "review_comment_message"
        ]
    ]
    sentiment_df.columns = ['product_category', 'review_score', 'review']
    # Remove rows without review text
    sentiment_df = sentiment_df.dropna(
        subset=["product_category", "review"]
    )

    return sentiment_df
print('functions are fine')