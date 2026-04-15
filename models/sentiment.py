def convert_sentiment(score):
    return 0 if score <= 2 else 1
    
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import re
import unidecode

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

    sentiment_df["label"] = sentiment_df["review_score"].apply(
        convert_sentiment
    )

    sentiment_df["clean_review"] = sentiment_df["review"].apply(
        clean_portuguese_text
    )

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
            ngram_range=(1,2),
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

df = pd.read_csv('ML_datasets/sentiment_df.csv')
train_sentiment_model(df)