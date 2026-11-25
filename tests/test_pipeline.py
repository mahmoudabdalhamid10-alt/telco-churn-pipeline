import numpy as np
import pandas as pd

from churn.pipeline import best_threshold, expected_cost, load, models


def test_load_handles_blank_total_charges():
    X, y = load()
    assert X["TotalCharges"].isna().sum() == 11
    assert len(X) == len(y) == 7043 and set(y) == {0, 1}
    assert "customerID" not in X.columns


def test_pipeline_fits_on_unseen_category():
    X, y = load()
    m = models(X)["logistic"].fit(X.iloc[:500], y.iloc[:500])
    row = X.iloc[[600]].copy()
    row["Contract"] = "Never seen"
    assert 0 <= m.predict_proba(row)[0, 1] <= 1


def test_expected_cost_counts_errors():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.1])
    assert expected_cost(y, p, 0.5, cost_fn=5, cost_fp=1) == 5 + 1


def test_threshold_is_lower_when_misses_are_expensive():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 2000), 0, 1)
    assert best_threshold(y, p, 10, 1) < best_threshold(y, p, 1, 10)
