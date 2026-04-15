# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# def prepare_churn_data(churn_df):

#     X = churn_df.drop(["customer_unique_id", "churn_label"], axis=1)
#     y = churn_df["churn_label"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test
# X_train, X_test, y_train, y_test = prepare_churn_data(df)


# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import roc_auc_score, f1_score

# def train_churn_models(X_train, X_test, y_train, y_test):

#     models = {
#         "Logistic": LogisticRegression(max_iter=500),
#         "RandomForest": RandomForestClassifier(n_estimators=200),
#         "XGBoost": XGBClassifier(n_estimators=200)
#     }

#     results = {}

#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         preds = model.predict(X_test)
#         prob = model.predict_proba(X_test)[:, 1]

#         results[name] = {
#             "ROC_AUC": roc_auc_score(y_test, prob),
#             "F1": f1_score(y_test, preds)
#         }

#     return results
# result = train_churn_models(X_train, X_test, y_train, y_test)
# print(result)


import pandas as pd
df = pd.read_csv('ML_datasets/churn_df.csv')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_churn_data(churn_df):

    X = churn_df.drop(["customer_unique_id", "churn_label"], axis=1)
    y = churn_df["churn_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepare_churn_data(df)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score

def train_churn_models(X_train, X_test, y_train, y_test):

    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    scale_pos_weight = neg / pos

    models = {
        "Logistic": LogisticRegression(
            max_iter=500,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss"
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "ROC_AUC": roc_auc_score(y_test, prob),
            "F1": f1_score(y_test, preds)
        }

    return results

print(train_churn_models(X_train, X_test, y_train, y_test))