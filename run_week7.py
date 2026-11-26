"""Run all Week 7 models without changing config.yaml."""

import copy

import main as pipeline_main


MODELS = {
    "dummy": {
        "strategy": "most_frequent",
    },
    "logistic_regression": {
        "max_iter": 2000,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "naive_bayes": {},
    "decision_tree": {
        "random_state": 42,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 200,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    },
    "oblique_random_forest": {
        "n_estimators": 200,
        "n_components": 10,
        "random_state": 42,
        "n_jobs": -1,
    },
    "adaboost": {
        "max_depth": 1,
        "n_estimators": 200,
        "learning_rate": 1.0,
        "random_state": 42,
    },
    "gradient_boosting": {
        "random_state": 42,
    },
    "hist_gradient_boosting": {
        "random_state": 42,
        "class_weight": "balanced",
    },
}


def main():
    original_loader = pipeline_main.load_config
    base_config = original_loader()

    # Check all model search spaces before starting.
    spaces = base_config["tuning"]["search_spaces"]
    missing = [
        name
        for name in MODELS
        if name != "dummy" and name not in spaces
    ]
    if missing:
        raise ValueError(f"Missing search spaces: {missing}")

    try:
        for number, (name, params) in enumerate(
            MODELS.items(), start=1
        ):
            config = copy.deepcopy(base_config)
            config["model"] = {
                "type": name,
                "params": params,
            }
            config["tuning"]["enabled"] = name != "dummy"
            config["tuning"]["final_tuning_only"] = False
            config["tuning"]["n_trials"] = 30

            # Keep each model's configuration in memory only.
            pipeline_main.load_config = (
                lambda path="config.yaml", current=config:
                copy.deepcopy(current)
            )

            print(
                f"\n{'=' * 60}\n"
                f"MODEL {number}/{len(MODELS)}: {name}\n"
                f"{'=' * 60}",
                flush=True,
            )

            pipeline_main.main()

        print(
            "\nALL WEEK 7 MODELS COMPLETED.\n"
            "Results: results/runs.csv",
            flush=True,
        )
    finally:
        pipeline_main.load_config = original_loader


if __name__ == "__main__":
    main()