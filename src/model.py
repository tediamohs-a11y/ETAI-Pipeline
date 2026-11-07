"""Model construction."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor


def oblique_random_forest(
    n_estimators: int = 100,
    n_components: int = 10,
    density="auto",
    random_state=None,
    n_jobs=None,
    **tree_params,
):
    """Build a forest of trees using random feature projections."""
    tree_params.setdefault("max_features", "sqrt")

    oblique_tree = Pipeline([
        (
            "project",
            SparseRandomProjection(
                n_components=n_components,
                density=density,
            ),
        ),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])

    return BaggingClassifier(
        estimator=oblique_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def adaboost(max_depth: int = 1, **params):
    """Build AdaBoost with decision trees of the chosen depth."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth),
        **params,
    )


_MODELS = {
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,
    "adaboost": adaboost,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Options: {list(_MODELS)}"
        )

    return _MODELS[model_type](**params)


def build_pipeline(
    preprocessing_config: dict,
    model_config: dict,
) -> Pipeline:
    return Pipeline([
        ("prep", build_preprocessor(preprocessing_config)),
        ("model", build_model(model_config)),
    ])

    