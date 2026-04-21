import pandas as pd
import re
import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.db import get_db_engine
from datetime import datetime


def convert_sentiment(score):
    return 0 if score <= 2 else 1


def clean_portuguese_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unidecode.unidecode(text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def train_sentiment_model(sentiment_df):
    sentiment_df = sentiment_df.copy()

    # Drop missing reviews
    sentiment_df = sentiment_df.dropna(subset=["review"])

    # Ensure string type
    sentiment_df["review"] = sentiment_df["review"].astype(str)

    sentiment_df["label"] = sentiment_df["review_score"].apply(convert_sentiment)

    sentiment_df["clean_review"] = sentiment_df["review"].apply(clean_portuguese_text)

    X_train, X_test, y_train, y_test = train_test_split(
        sentiment_df["clean_review"],
        sentiment_df["label"],
        test_size=0.2,
        random_state=42,
        stratify=sentiment_df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=5
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(classification_report(y_test, preds))

    return pipeline


def save_sentiment_to_db(sentiment_df, pipeline):
    """
    Predicts sentiment for all reviews, aggregates them by category,
    and saves the summary to category_sentiment_summary table.
    """
    engine = get_db_engine()

    # Predict all data
    sentiment_df = sentiment_df.dropna(subset=["review"]).copy()
    sentiment_df["clean_review"] = sentiment_df["review"].astype(str).apply(clean_portuguese_text)
    sentiment_df['predicted_sentiment'] = pipeline.predict(sentiment_df["clean_review"])

    summary = sentiment_df.groupby('product_category').agg(
        avg_review_score=('review_score', 'mean'),
        review_count=('review', 'count')
    ).reset_index()

    summary['last_updated'] = datetime.now()

    try:
        summary.to_sql("category_sentiment_summary", engine, if_exists="replace", index=False)
        print(f"Successfully saved sentiment summary for {len(summary)} categories.")
    except Exception as e:
        print(f"Error saving to DB: {e}")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/ML_datasets/sentiment_df.csv')
    df = pd.read_csv(data_path)
    pipeline = train_sentiment_model(df)
    save_sentiment_to_db(df, pipeline)