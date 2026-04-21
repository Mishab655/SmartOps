import os
import pandas as pd
from backend.core.db import get_db_engine


class DecisionEngine:
    def __init__(self, db_engine=None):
        """
        Uses the provided SQLAlchemy engine to connect to PostgreSQL.
        If none provided, uses db_utils.
        """
        self.db_engine = db_engine or get_db_engine()

    def load_data(self):
        """
        Loads all required data directly from PostgreSQL.
        No CSV reads — fully DB-driven .
        """
        try:
            self.forecast_df = pd.read_sql("SELECT * FROM category_forecast", self.db_engine)
            self.sentiment_df = pd.read_sql("SELECT * FROM category_sentiment_summary", self.db_engine)
            self.churn_df = pd.read_sql("SELECT * FROM customer_churn_prediction", self.db_engine)
        except Exception as e:
            print(f"Error reading from DB. Did you run the models yet? Error: {e}")
            raise

    def _get_customer_affinities(self):
        """
        Identifies the most frequently purchased product category for each customer.
        Queries PostgreSQL directly
        """
        affinity_query = """
            SELECT
                c.customer_unique_id,
                p.product_category_name AS product_category,
                COUNT(oi.order_id) AS purchase_count
            FROM olist_orders o
            JOIN olist_customers c ON o.customer_id = c.customer_id
            JOIN olist_order_items oi ON o.order_id = oi.order_id
            JOIN olist_products p ON oi.product_id = p.product_id
            WHERE p.product_category_name IS NOT NULL
            GROUP BY c.customer_unique_id, p.product_category_name
        """
        try:
            affinity = pd.read_sql(affinity_query, self.db_engine)
        except Exception as e:
            print(f"Warning: Could not load customer affinities from DB: {e}")
            print("Falling back to empty affinities. "
                  "Load raw Olist tables into the DB to enable customer-level actions.")
            return pd.DataFrame(columns=["customer_unique_id", "product_category", "purchase_count"])

        # Get top category per customer
        idx = affinity.groupby("customer_unique_id")["purchase_count"].idxmax()
        top_affinity = affinity.loc[idx]

        return top_affinity

    def generate_category_insights(self):
        """
        Combines forecasting and sentiment to generate category-level actions.
        """
        cat_sentiment = self.sentiment_df.copy()

        # Aggregate predicted sales across all future dates
        cat_forecast = self.forecast_df.groupby("product_category").agg(
            predicted_sales=("predicted_sales", "sum")
        ).reset_index()

        # Merge metrics
        category_health = cat_forecast.merge(cat_sentiment, on="product_category", how="inner")

        # Define thresholds
        high_sales_threshold = category_health["predicted_sales"].quantile(0.75)
        low_sales_threshold = category_health["predicted_sales"].quantile(0.25)
        high_sentiment_threshold = 4.0
        low_sentiment_threshold = 3.5

        actions = []

        for _, row in category_health.iterrows():
            cat = row["product_category"]
            sales = row["predicted_sales"]
            sentiment = row["avg_review_score"]

            if sales >= high_sales_threshold and sentiment >= high_sentiment_threshold:
                actions.append({
                    "action_type": "Scale Operations & Highlight",
                    "target_entity_id": cat,
                    "entity_type": "category",
                    "action_description": (
                        "Star Category: High forecasted demand and high sentiment. "
                        "Increase marketing budget, assign premium placement, and ensure supply chain."
                    ),
                    "priority": 1
                })
            elif low_sales_threshold <= sales < high_sales_threshold and sentiment >= 3.5:
                actions.append({
                    "action_type": "Maintain & Optimize",
                    "target_entity_id": cat,
                    "entity_type": "category",
                    "action_description": (
                        "Cash Cow: Steady demand and good satisfaction. "
                        "Keep stock levels steady. No deep discounts needed."
                    ),
                    "priority": 3
                })
            elif sales >= high_sales_threshold and sentiment < low_sentiment_threshold:
                actions.append({
                    "action_type": "Critical QA / Stop Sale Investigaton",
                    "target_entity_id": cat,
                    "entity_type": "category",
                    "action_description": (
                        f"Quality Crisis: High demand but low sentiment ({sentiment:.1f}/5). "
                        "Urgent QA check needed to prevent returns."
                    ),
                    "priority": 1
                })
            elif sales < low_sales_threshold and sentiment >= high_sentiment_threshold:
                actions.append({
                    "action_type": "Awareness Campaign / Bundle Offers",
                    "target_entity_id": cat,
                    "entity_type": "category",
                    "action_description": (
                        "Hidden Gem: Low demand but high sentiment. "
                        "Promote via targeted email campaigns or bundle with 'Stars'."
                    ),
                    "priority": 2
                })
            elif sales < low_sales_threshold and sentiment < low_sentiment_threshold:
                actions.append({
                    "action_type": "Sunset & Liquidate",
                    "target_entity_id": cat,
                    "entity_type": "category",
                    "action_description": (
                        "Dead Weight: Poor demand and poor sentiment. "
                        "Consider heavily discounting to clear inventory and phasing out."
                    ),
                    "priority": 2
                })

        return pd.DataFrame(actions)

    def generate_customer_actions(self, category_health_df):
        """
        Combines churn prediction with category affinities to create personalized offers.
        """
        affinities = self._get_customer_affinities()

        if affinities.empty:
            print("No customer affinity data available. Skipping customer-level actions.")
            return pd.DataFrame()

        # Link All Customers to their Favourite Category
        customer_profiles = self.churn_df.merge(affinities, on="customer_unique_id", how="inner")

        # Find high sentiment categories to target for personalized offers
        high_sentiment_cats = (
            category_health_df[category_health_df["avg_review_score"] >= 4.0]["product_category"].tolist()
        )

        actions = []

        for _, row in customer_profiles.iterrows():
            cust_id = row["customer_unique_id"]
            fav_cat = row["product_category"]
            risk = row["churn_risk"]

            if risk == "Low":
                actions.append({
                    "action_type": "VIP Engagement",
                    "target_entity_id": cust_id,
                    "entity_type": "customer",
                    "action_description": (
                         f"Loyal Champion. Send 'Thank You' rewards (e.g., VIP points, early access) based on their favorite category: {fav_cat}."
                    ),
                    "priority": 2
                })
            elif risk == "Medium":
                actions.append({
                    "action_type": "Re-engagement Nudge",
                    "target_entity_id": cust_id,
                    "entity_type": "customer",
                    "action_description": (
                        f"At-Risk Customer. Send a moderate discount (10%) or personalized recommendation for their top category: {fav_cat}."
                    ),
                    "priority": 1
                })
            elif risk == "High":
                if fav_cat in high_sentiment_cats:
                    actions.append({
                        "action_type": "Aggressive Win-Back",
                        "target_entity_id": cust_id,
                        "entity_type": "customer",
                        "action_description": (
                            f"Customer churned but loved a high-quality category. "
                            f"Send 20-30% discount offer for favourite 'Star' category: {fav_cat}."
                        ),
                        "priority": 1
                    })
                else:
                    actions.append({
                        "action_type": "Exploratory Win-Back",
                        "target_entity_id": cust_id,
                        "entity_type": "customer",
                        "action_description": (
                            "Customer churned. Previous category had low sentiment. "
                            "Send promotional offer for generic 'Star' categories across the store."
                        ),
                        "priority": 2
                    })

        return pd.DataFrame(actions)

    def run_engine(self):
        """
        Executes all decision logic and returns a compiled DataFrame of actions.
        """
        print("Loading data from DB...")
        self.load_data()

        print("Generating Category-level insights...")
        category_actions = self.generate_category_insights()

        # Re-build category health for customer linking
        cat_sentiment = self.sentiment_df[["product_category", "avg_review_score"]].copy()
        cat_forecast = (
            self.forecast_df.groupby("product_category")["predicted_sales"].sum().reset_index()
        )
        category_health_df = cat_forecast.merge(cat_sentiment, on="product_category")

        print("Generating Customer-level actions...")
        customer_actions = self.generate_customer_actions(category_health_df)

        # Combine all actionable insights
        all_actions = pd.concat([category_actions, customer_actions], ignore_index=True)

        print(f"Generated {len(all_actions)} actionable business rules.")
        return all_actions


if __name__ == "__main__":
    engine = DecisionEngine()
    actions_df = engine.run_engine()

    # Save to PostgreSQL decision_action_log
    try:
        actions_df.to_sql("3", engine.db_engine, if_exists="append", index=False)
        print("Saved decisions to PostgreSQL table: decision_action_log")
    except Exception as e:
        print(f"Failed to write to DB: {e}")

    print("\nSample Actions:")
    print(actions_df.head(10))
