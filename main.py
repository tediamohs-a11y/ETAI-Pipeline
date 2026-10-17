"""Run the predictive pipeline with optional week-5 tuning."""

import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    split_dev_test,
)
from src.model import build_pipeline
from src.evaluate import (
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
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Fixed cleaning rules: nothing is learned here.
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(
        df_clean, config["diagnostics"].get("id_column")
    )

    mnar_sources = config["preprocessing"].get(
        "mnar_indicator_sources", []
    )
    X, y, extras = split_features_target(
        df_clean, config["data"], mnar_sources
    )

    # The final test set remains locked.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipeline = build_pipeline(
        config["preprocessing"], config["model"]
    )

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # 1. Evaluate the model before tuning.
    print("\n=== Default-hyperparameter CV ===")
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs
    )

    report = "Default-hyperparameter CV\n"
    report += cv_report(fold_scores, scoring)

    tuning_config = config.get("tuning", {})

    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}

        if model_type not in search_spaces:
            raise ValueError(
                f"No tuning search space for '{model_type}'. "
                "Add it to tuning.search_spaces in config.yaml."
            )

        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]

        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        # 2. Honest evaluation of the complete tuning procedure.
        print("\n=== Nested cross-validation ===")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev,
            cv, inner_cv, scoring, search_space,
            n_trials, tuning_seed, n_jobs=n_jobs,
        )

        report += "\n\nNested cross-validation\n"
        report += cv_report(nested_scores, scoring)

        # 3. Choose final parameters using all development rows.
        print("\n=== Tuning on all development rows ===")
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev,
            inner_cv, scoring, search_space,
            n_trials, tuning_seed, n_jobs=n_jobs,
        )

        report += "\n\n" + tuning_report(
            study, nested_scores, scoring
        )

    # When tuning is enabled, these are nested OOF predictions.
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev,
        y_oof,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    # Fit the selected pipeline on all development rows.
    final_model = pipeline.fit(X_dev, y_dev)

    refit = (
        f"Final model: {config['model']['type']} "
        f"refit on all {len(X_dev)} development rows."
    )
    print(refit)
    report += "\n" + refit + "\n"

    locked = (
        f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
        f"Development set: {len(X_dev)} rows."
    )
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get(
        "results_dir", "results"
    )
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()