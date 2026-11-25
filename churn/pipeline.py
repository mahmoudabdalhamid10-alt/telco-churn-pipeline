import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"


URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load(path="data/telco.csv"):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    df = pd.read_csv(path)
    # 11 rows have a blank TotalCharges (tenure == 0, brand-new customers): make it NaN, impute in the pipeline.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET, "customerID"])
    return X, y


def categorical_columns(X):
    return [c for c in X.columns if c not in NUMERIC]


def preprocessor(X, scale=True):
    num = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns(X)),
    ])


def models(X):
    """All preprocessing lives inside the pipeline so cross-validation never sees test folds."""
    return {
        "logistic": Pipeline([("prep", preprocessor(X)), ("clf", LogisticRegression(max_iter=2000, C=0.5))]),
        "random_forest": Pipeline([("prep", preprocessor(X, scale=False)),
                                   ("clf", RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=0))]),
        "hist_gb": Pipeline([("prep", preprocessor(X, scale=False)),
                             ("clf", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                    random_state=0))]),
    }


def expected_cost(y_true, p, threshold, cost_fn, cost_fp):
    pred = p >= threshold
    fn = ((~pred) & (y_true == 1)).sum()
    fp = (pred & (y_true == 0)).sum()
    return fn * cost_fn + fp * cost_fp


def best_threshold(y_true, p, cost_fn, cost_fp):
    grid = np.linspace(0.02, 0.98, 97)
    costs = [expected_cost(y_true, p, t, cost_fn, cost_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])
