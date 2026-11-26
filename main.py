"""Run the Week 7 pipeline with multiple metrics and a scoreboard."""

from datetime import datetime

import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    add_engineered_features,
    split_features_target,
    split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    oof_classification_report,
    fairness_report,
)
from src.tuning import (
    tune_pipeline,
    nested_cross_validate,
    tuning_report,
)
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    # Fixed cleaning rules and formulas learn nothing from the data.
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(
        df_clean,
        config["diagnostics"].get("id_column"),
    )
    df_clean = add_engineered_features(df_clean)

    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )
    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        mnar_sources,
    )

    # Preserve the original locked test set.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = (
        split_dev_test(
            X,
            y,
            extras,
            test_size=config["test_set"]["size"],
            random_state=config["test_set"]["random_state"],
        )
    )

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)

    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=(
            cv_config.get("random_state") if shuffle else None
        ),
    )

    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]

    if selection_metric not in metrics:
        raise ValueError(
            "cv.selection_metric must also appear in cv.metrics."
        )

    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    if tuning_config.get("final_tuning_only", False):
        raise ValueError(
            "Set tuning.final_tuning_only to false for Week 7. "
            "The scoreboard requires full CV or nested CV."
        )

    search_spaces = tuning_config.get("search_spaces") or {}

    if tuning_enabled and model_type not in search_spaces:
        raise ValueError(
            f"No tuning search space for '{model_type}'. "
            "Add it or set tuning.enabled to false."
        )

    print(
        f"Model: {model_type} | "
        f"Selection metric: {selection_metric} | "
        f"{cv_config['n_splits']}-fold CV",
        flush=True,
    )
    print(
        f"Development rows: {len(X_dev)} | "
        f"Locked test rows: {len(X_test)} (not evaluated)",
        flush=True,
    )

    pipeline = build_pipeline(
        config["preprocessing"], config["model"]
    )
    floor_pipeline = build_pipeline(
        config["preprocessing"],
        {"type": "dummy", "params": {"strategy": "most_frequent"}},
    )

    # No-model floor, evaluated on exactly the same folds.
    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline,
        X_dev,
        y_dev,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs=n_jobs,
    )
    floor = floor_scores["validation"].mean()

    # Evaluate the parameters currently specified in config.yaml.
    print("\n=== Configured parameters: CV ===", flush=True)
    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline,
        X_dev,
        y_dev,
        cv,
        scorers,
        selection_metric,
        n_jobs=n_jobs,
    )
    untuned = fold_scores["validation"].mean()

    default_report = cv_report(
        fold_scores,
        metrics,
        selection_metric,
        "Configured parameters: cross-validation",
    )
    print(default_report)

    report = (
        f"Configured parameters: "
        f"{config['model'].get('params')}\n\n"
        + default_report
        + f"\n\nDummy floor: {selection_metric} = {floor:.3f}"
    )

    final_scores = fold_scores
    study = None

    if tuning_enabled:
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        print(
            f"\n=== Nested CV: {n_trials} trials per study ===",
            flush=True,
        )
        final_scores, oof_proba = nested_cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scorers,
            selection_metric,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )

        nested_report = cv_report(
            final_scores,
            metrics,
            selection_metric,
            "Nested cross-validation",
        )
        print(nested_report)
        report += "\n\n" + nested_report

        # Save completed nested results before final tuning.
        results_dir = config.get("output", {}).get(
            "results_dir", "results"
        )
        checkpoint_path = save_run(
            results_dir,
            config,
            report
            + "\n\nCHECKPOINT: final tuning and refit still pending.\n",
            run=run + "_nested_checkpoint",
        )
        print(
            f"\nNested-CV checkpoint saved to {checkpoint_path}",
            flush=True,
        )

        print(
            "\n=== Final tuning on all development rows ===",
            flush=True,
        )
        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )

        tuning_text = tuning_report(
            study, final_scores, selection_metric
        )
        print(tuning_text)
        report += "\n\n" + tuning_text

    classification_text = oof_classification_report(
        y_dev, oof_proba
    )
    fairness_text = fairness_report(
        y_dev,
        oof_proba,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    print("\n" + classification_text)
    print("\n" + fairness_text)
    report += "\n\n" + classification_text
    report += "\n\n" + fairness_text

    pipeline.fit(X_dev, y_dev)

    refit = (
        f"Final model: {model_type} refit on "
        f"{len(X_dev)} development rows."
    )
    locked = (
        f"Locked test set: {len(X_test)} rows, not evaluated."
    )
    report += "\n\n" + refit + "\n" + locked + "\n"

    row = {
        "run": run,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": tuning_enabled,
        "n_trials": (
            tuning_config["n_trials"] if tuning_enabled else 0
        ),
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "numeric_features": str(
            config["preprocessing"]["numeric_features"]
        ),
        "params": str(config["model"].get("params")),
        "tuned_params": (
            str(study.best_params) if study is not None else ""
        ),
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
    }

    for fold, score in enumerate(
        final_scores["validation"], start=1
    ):
        row[f"fold_{fold}"] = score

    for name in metrics:
        row[f"{name}_mean"] = final_scores[name].mean()
        row[f"{name}_std"] = final_scores[name].std(ddof=1)

    output = config.get("output", {})
    scoreboard_path = output.get(
        "scoreboard", "results/runs.csv"
    )
    runs = save_scores(scoreboard_path, row)

    same = runs[
        (runs["data"] == row["data"])
        & (runs["metric"] == selection_metric)
        & (runs["encoder"] == row["encoder"])
        & (runs["scaler"] == row["scaler"])
        & (runs["numeric_features"] == row["numeric_features"])
    ]

    board = (
        same.sort_values("mean", ascending=False)[
            [
                "run",
                "model",
                "tuned",
                "n_trials",
                "mean",
                "std",
                "untuned",
                "floor",
            ]
        ]
        .rename(columns={"mean": selection_metric})
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\n=== Scoreboard ===\n" + board)
    report += "\nScoreboard:\n" + board + "\n"

    path = save_run(
        output.get("results_dir", "results"),
        config,
        report,
        run=run,
    )

    print(f"\n{refit}\n{locked}")
    print(f"Full report saved to {path}")
    print(f"Scoreboard saved to {scoreboard_path}")


if __name__ == "__main__":
    main()