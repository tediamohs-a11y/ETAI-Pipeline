"""Hyperparameter tuning and nested CV with multiple metrics."""

import numpy as np
import pandas as pd
import optuna

from sklearn.base import clone
from sklearn.model_selection import cross_val_score


def tune_pipeline(
    pipeline,
    X,
    y,
    cv,
    scorer,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Tune using the selection metric; return an unfitted pipeline."""

    def objective(trial):
        params = {}

        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(
                    name,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    spec["choices"],
                )
            else:
                raise ValueError(
                    f"Unknown search-space type for {name}: "
                    f"{spec['type']}. "
                    "Options: int, float, categorical"
                )

        candidate = clone(pipeline).set_params(**params)

        scores = cross_val_score(
            candidate,
            X,
            y,
            cv=cv,
            scoring=scorer,
            n_jobs=n_jobs,
        )

        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)

    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(
    pipeline,
    X,
    y,
    outer_cv,
    inner_cv,
    scorers: dict,
    selection_metric: str,
    search_space: dict,
    n_trials: int,
    random_state: int,
    n_jobs: int = 1,
):
    """Tune inside each outer training fold and evaluate every metric."""
    rows = []
    oof_proba = np.empty(len(X))

    for fold, (train_idx, val_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(
            pipeline,
            X_train,
            y_train,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            random_state,
            n_jobs=n_jobs,
        )

        best_pipeline.fit(X_train, y_train)

        train_score = scorers[selection_metric](
            best_pipeline, X_train, y_train
        )

        val_scores = {
            name: scorer(best_pipeline, X_val, y_val)
            for name, scorer in scorers.items()
        }

        oof_proba[val_idx] = best_pipeline.predict_proba(
            X_val
        )[:, 1]

        rows.append({
            "fold": fold,
            "train": train_score,
            "validation": val_scores[selection_metric],
            "gap": train_score - val_scores[selection_metric],
            **val_scores,
            "inner_best": study.best_value,
            **study.best_params,
        })

    return pd.DataFrame(rows), oof_proba


def tuning_report(
    study,
    nested_scores: pd.DataFrame,
    selection_metric: str,
    top: int = 5,
) -> str:
    """Return final tuning results and the nested-CV optimism check."""
    trials = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs")
    )

    trials = trials.rename(
        columns=lambda column: column.replace(
            "params_", ""
        ).replace("user_attrs_", "")
    )

    score_column = f"mean {selection_metric}"
    trials = trials.rename(
        columns={"number": "trial", "value": score_column}
    )

    inner = nested_scores["inner_best"].mean()
    outer = nested_scores["validation"].mean()

    top_trials = (
        trials.sort_values(score_column, ascending=False)
        .head(top)
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    lines = [
        (
            "Tuning on the whole development set "
            f"({len(study.trials)} Optuna trials, "
            f"metric: {selection_metric})"
        ),
        f"Best hyperparameters: {study.best_params}",
        (
            f"Best mean CV score:   {study.best_value:.3f}  "
            "(the maximum over all trials -- optimistic)"
        ),
        "",
        f"Top {top} trials:",
        top_trials,
        "",
        "Optimism check (nested CV):",
        (
            "  tuning score inside the outer folds "
            f"(inner_best) mean = {inner:.3f}"
        ),
        (
            "  honest score on the outer folds "
            f"(validation)     mean = {outer:.3f}"
        ),
        (
            f"  optimism = {inner - outer:+.3f}"
            "   -> report the honest one"
        ),
    ]

    return "\n".join(lines)