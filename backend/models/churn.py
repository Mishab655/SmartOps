"""
Churn Risk Analysis using RFM Scoring

Approach: Unsupervised RFM segmentation instead of a binary classifier.

Why: Olist data is structurally one-purchase-dominant. ~95% of customers
never repeat, making any binary churn label overwhelmingly imbalanced.
A supervised classifier on such data essentially learns a constant function.

Instead, we:
  1. Build RFM features (Recency, Frequency, Monetary) for repeat customers.
  2. Score each dimension on a 1-3 quantile scale.
  3. Derive a composite rfm_risk_score (3-9) and assign a churn_risk band:
       - High   (score 3-5): Customer is inactive, low value, infrequent
       - Medium (score 6-7): Customer is moderately engaged
       - Low    (score 8-9): Recent, frequent, high-value loyal customer
  4. Display segment distribution and export the scored dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.db import get_db_engine
from datetime import datetime


def load_and_validate(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ML_datasets/churn_df.csv')
    df = pd.read_csv(filepath)

    required_cols = {
        "customer_unique_id", "recency", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_risk_score", "churn_risk"
    }
    assert required_cols.issubset(df.columns), (
        "churn_df.csv is missing expected RFM columns. Re-run data_prep.ipynb."
    )
    return df


def print_segment_summary(df):
    print("=" * 50)
    print("RFM Churn Risk Segment Distribution")
    print("=" * 50)

    segment_summary = (
        df.groupby("churn_risk", observed=True)
        .agg(
            customer_count=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            avg_rfm_score=("rfm_risk_score", "mean")
        )
        .reset_index()
    )

    segment_order = ["High", "Medium", "Low"]
    segment_summary["churn_risk"] = pd.Categorical(
        segment_summary["churn_risk"], categories=segment_order, ordered=True
    )
    segment_summary = segment_summary.sort_values("churn_risk")
    print(segment_summary.to_string(index=False))

    print("\nRFM Score Statistics:")
    print(df[["r_score", "f_score", "m_score", "rfm_risk_score"]].describe().round(2))

    return segment_summary


def plot_churn_analysis(df, segment_summary):
    segment_order = ["High", "Medium", "Low"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("RFM Churn Risk Analysis", fontsize=14, fontweight="bold")

    # 1. Churn Risk Band Distribution
    risk_counts = df["churn_risk"].value_counts()[segment_order]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    axes[0].bar(risk_counts.index, risk_counts.values, color=colors)
    axes[0].set_title("Churn Risk Band Distribution")
    axes[0].set_xlabel("Churn Risk")
    axes[0].set_ylabel("Customer Count")
    for i, v in enumerate(risk_counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontsize=9)

    # 2. Composite RFM Score Histogram
    axes[1].hist(df["rfm_risk_score"], bins=range(3, 11), color="#3498db",
                 edgecolor="white", rwidth=0.8, align="left")
    axes[1].set_title("Composite RFM Risk Score Distribution")
    axes[1].set_xlabel("RFM Risk Score (3=High Risk, 9=Low Risk)")
    axes[1].set_ylabel("Customer Count")
    axes[1].set_xticks(range(3, 10))

    # 3. Average Monetary Value by Risk Band
    avg_monetary = segment_summary.set_index("churn_risk")["avg_monetary"][segment_order]
    axes[2].bar(avg_monetary.index, avg_monetary.values, color=colors)
    axes[2].set_title("Avg. Monetary Value by Churn Risk Band")
    axes[2].set_xlabel("Churn Risk")
    axes[2].set_ylabel("Avg. Total Spend (BRL)")

    plt.tight_layout()
    plt.show()


def save_churn_scores_to_db(scored_df, model_version="rfm_v1"):
    """
    Saves the computed RFM scores and risk bands to customer_churn_prediction.
    """
    engine = get_db_engine()

    db_df = scored_df[[
        "customer_unique_id", "recency", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_risk_score", "churn_risk"
    ]].copy()

    db_df["model_version"] = model_version
    db_df["scored_at"] = datetime.now()

    try:
        db_df.to_sql("customer_churn_prediction", engine, if_exists="replace", index=False)
        print(f"Successfully saved {len(db_df)} scored customers to DB.")
    except Exception as e:
        print(f"Error saving to DB: {e}")


if __name__ == "__main__":
    df = load_and_validate()

    segment_summary = print_segment_summary(df)
    plot_churn_analysis(df, segment_summary)

    save_churn_scores_to_db(df)
    print("\nDone. Use rfm_risk_score and churn_risk columns in the Decision Engine.")