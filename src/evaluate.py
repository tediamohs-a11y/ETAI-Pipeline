"""Evaluate the pipeline using multiple metrics and OOF probabilities."""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    get_scorer,
    make_scorer,
)
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """Build scorers for sklearn metrics and custom F-beta scores."""
    scorers = {}

    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])
            scorers[name] = make_scorer(
                fbeta_score,
                beta=beta,
                zero_division=0,
            )
        else:
            scorers[name] = get_scorer(name)

    return scorers


def cross_validate_pipeline(
    pipeline,
    X,
    y,
    cv,
    scorers: dict,
    selection_metric: str,
    n_jobs: int = 1,
):
    """Return per-fold metrics and OOF positive-class probabilities."""
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )

    fold_scores = pd.DataFrame({
        "fold": range(1, cv.get_n_splits() + 1),
    })

    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = (
        fold_scores["train"] - fold_scores["validation"]
    )

    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X))

    for model, val_idx in zip(
        scores["estimator"],
        scores["indices"]["test"],
    ):
        oof_proba[val_idx] = model.predict_proba(
            X.iloc[val_idx]
        )[:, 1]

    return fold_scores, oof_proba


def cv_report(
    fold_scores: pd.DataFrame,
    metrics: list,
    selection_metric: str,
    title: str,
) -> str:
    """Return per-fold scores and summaries for every metric."""
    lines = [
        (
            f"{title} ({len(fold_scores)} stratified folds, "
            f"selection metric: {selection_metric})"
        ),
        "",
        fold_scores.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        ),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(
            float_format=lambda value: f"{value:.3f}",
        ),
        (
            f"\n{selection_metric}: train-validation gap "
            f"mean = {fold_scores['gap'].mean():+.3f}"
        ),
    ]

    return "\n".join(lines)


def oof_classification_report(
    y_true,
    oof_proba,
    threshold: float = 0.5,
) -> str:
    """Return the OOF confusion matrix and classification report."""
    y_pred = (oof_proba >= threshold).astype(int)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["true 0", "true 1"],
        columns=["predicted 0", "predicted 1"],
    )

    return (
        "Out-of-fold predictions, development set "
        f"(threshold {threshold}):\n\n"
        f"{cm.to_string()}\n\n"
        f"{classification_report(y_true, y_pred, zero_division=0)}"
    )


def fairness_report(
    y_true,
    oof_proba,
    extras: pd.DataFrame,
    sensitive_attr: str = "race",
    threshold: float = 0.5,
) -> str:
    """Compare model and COMPAS false-positive rates by group."""
    df = extras.copy()

    df[sensitive_attr] = (
        df[sensitive_attr].astype(object).fillna("unknown")
    )
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = (
        df["score_text"] != "Low"
    ).astype(int)

    lines = [
        (
            f"False positive rate by {sensitive_attr} "
            "(development set, out-of-fold)"
        ),
        "(share of people who did NOT reoffend, "
        "but were predicted to)",
        "",
    ]

    comparisons = [
        ("Our model", "y_pred_model"),
        ("COMPAS's own score", "y_pred_compas"),
    ]

    for label, column in comparisons:
        lines.append(f"  {label}:")

        for group, group_data in df.groupby(sensitive_attr):
            negatives = group_data[group_data["y_true"] == 0]

            if len(negatives) == 0:
                continue

            fpr = (negatives[column] == 1).mean()
            lines.append(
                f"    {group:<20s} FPR = {fpr:.2f}  "
                f"(n={len(negatives)})"
            )

        lines.append("")

    return "\n".join(lines)