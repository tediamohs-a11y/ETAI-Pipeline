"""Save full reports and a CSV scoreboard of model runs."""

import os
from datetime import datetime

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Append one run to the CSV and return the complete scoreboard."""
    if os.path.exists(path):
        runs = pd.read_csv(path, dtype={"run": str})
    else:
        runs = pd.DataFrame()

    runs = pd.concat(
        [runs, pd.DataFrame([row])],
        ignore_index=True,
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)

    return runs


def save_run(
    results_dir: str,
    config: dict,
    report_text: str,
    run: str = None,
) -> str:
    """Save a full report using the same run ID as the scoreboard."""
    os.makedirs(results_dir, exist_ok=True)

    if run is None:
        run = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    preprocessing = config["preprocessing"]
    tuning = config.get("tuning", {})

    selection_metric = cv.get(
        "selection_metric",
        cv.get("scoring", "accuracy"),
    )
    metrics = cv.get("metrics", [selection_metric])

    header = (
        f"Run: {run}\n"
        f"Data: {config['data']['path']}\n"
        f"Model: {config['model']['type']}  "
        f"params={config['model'].get('params')}\n"
        f"Tuning enabled: {tuning.get('enabled', False)}\n"
        f"Tuning settings: {tuning}\n"
        f"Preprocessing: encoder={preprocessing['encoder']}  "
        f"scaler={preprocessing['scaler']}\n"
        f"Numeric features: "
        f"{preprocessing['numeric_features']}\n"
        f"Categorical features: "
        f"{preprocessing['categorical_features']}\n"
        f"CV: {cv['n_splits']} stratified folds  "
        f"shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Selection metric: {selection_metric}\n"
        f"Reported metrics: {metrics}\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  "
        "(not evaluated)\n"
        + "=" * 60
        + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(header + report_text)

    return path