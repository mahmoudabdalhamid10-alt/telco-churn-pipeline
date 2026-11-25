"""python -m churn.run  ->  results/*.csv, results/*.png"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .pipeline import best_threshold, expected_cost, load, models, preprocessor

# Assumption (see README): a missed churner costs 5x a wasted retention offer.
COST_FN, COST_FP = 5.0, 1.0
SEED = 0


def main():
    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows = []
    cands = {"baseline (prior)": Pipeline([("prep", preprocessor(X)), ("clf", DummyClassifier(strategy="prior"))])}
    cands.update(models(X))
    for name, m in cands.items():
        r = cross_validate(m, Xtr, ytr, cv=cv, scoring=["roc_auc", "average_precision", "neg_brier_score"])
        rows.append(dict(model=name,
                         cv_roc_auc=r["test_roc_auc"].mean(), cv_roc_auc_std=r["test_roc_auc"].std(),
                         cv_pr_auc=r["test_average_precision"].mean(),
                         cv_brier=-r["test_neg_brier_score"].mean()))
    cvdf = pd.DataFrame(rows)
    cvdf.to_csv("results/cv_models.csv", index=False)
    print(cvdf.round(4).to_string(index=False))

    # Model choice by cross-validated PR-AUC on the training set only.
    best = cvdf[cvdf.model != "baseline (prior)"].sort_values("cv_pr_auc", ascending=False).iloc[0].model
    model = models(X)[best]
    # Threshold chosen on out-of-fold training predictions, never on the test set.
    oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
    thr = best_threshold(ytr.values, oof, COST_FN, COST_FP)

    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    yt = yte.values
    out = dict(
        model=best, threshold=thr, cost_fn=COST_FN, cost_fp=COST_FP,
        test_roc_auc=roc_auc_score(yt, p), test_pr_auc=average_precision_score(yt, p),
        test_brier=brier_score_loss(yt, p), test_churn_rate=float(yt.mean()),
        cost_at_threshold=float(expected_cost(yt, p, thr, COST_FN, COST_FP)),
        cost_at_0_5=float(expected_cost(yt, p, 0.5, COST_FN, COST_FP)),
        cost_flag_everyone=float(expected_cost(yt, np.ones_like(p), 0.5, COST_FN, COST_FP)),
        cost_flag_nobody=float(expected_cost(yt, np.zeros_like(p), 0.5, COST_FN, COST_FP)),
        recall_at_threshold=float(((p >= thr) & (yt == 1)).sum() / (yt == 1).sum()),
        precision_at_threshold=float(((p >= thr) & (yt == 1)).sum() / max((p >= thr).sum(), 1)),
        n_test=int(len(yt)),
    )
    json.dump(out, open("results/test_metrics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

    frac, mean = calibration_curve(yt, p, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].plot(mean, frac, "o-")
    ax[0].set(xlabel="Predicted churn probability", ylabel="Observed churn rate", title=f"Calibration ({best}, test set)")
    grid = np.linspace(0.02, 0.98, 97)
    ax[1].plot(grid, [expected_cost(yt, p, t, COST_FN, COST_FP) for t in grid])
    ax[1].axvline(thr, color="red", ls="--", label=f"chosen threshold {thr:.2f}")
    ax[1].set(xlabel="Decision threshold", ylabel="Cost (units of one wasted offer)", title="Cost vs threshold (test set)")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("results/calibration_cost.svg")

    imp = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=10, random_state=SEED)
    idf = pd.DataFrame({"feature": Xte.columns, "importance": imp.importances_mean, "std": imp.importances_std})
    idf = idf.sort_values("importance", ascending=False)
    idf.to_csv("results/permutation_importance.csv", index=False)
    print(idf.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
