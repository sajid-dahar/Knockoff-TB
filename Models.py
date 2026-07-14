"""Machine learning models for TB resistance prediction."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


MODEL_ALIASES = {
    "catboost": "catboost",
    "catb": "catboost",
    "lightgbm": "lightgbm",
    "ligb": "lightgbm",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "gbdt": "gbdt",
    "rf": "rf",
}


def build_classifier(name: str, random_state: int = 2025):
    name = MODEL_ALIASES.get(name, name)
    if name == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("catboost not installed")
        return CatBoostClassifier(
            random_state=random_state,
            verbose=0,
            auto_class_weights="Balanced",
            iterations=300,
            depth=6,
            learning_rate=0.05,
        )
    if name == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm not installed")
        return LGBMClassifier(
            random_state=random_state,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            class_weight="balanced",
            verbosity=-1,
        )
    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        return XGBClassifier(
            random_state=random_state,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            eval_metric="logloss",
        )
    if name == "gbdt":
        return GradientBoostingClassifier(random_state=random_state, max_depth=4)
    if name == "rf":
        return RandomForestClassifier(
            random_state=random_state,
            n_estimators=300,
            class_weight="balanced_subsample",
            max_depth=12,
        )
    raise ValueError(f"Unknown model: {name}")


def fit_classifier(model, x: np.ndarray, y: np.ndarray):
    model.fit(x, y)
    return model


def tune_classifier(name: str, x: np.ndarray, y: np.ndarray, cv_folds: int = 5, random_state: int = 2025):
    name = MODEL_ALIASES.get(name, name)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if name == "catboost":
        pipe = build_classifier(name, random_state)
        grid = {"depth": [4, 6], "learning_rate": [0.03, 0.05], "iterations": [200, 400]}
    elif name == "lightgbm":
        pipe = build_classifier(name, random_state)
        grid = {"max_depth": [4, 6], "learning_rate": [0.03, 0.05], "n_estimators": [200, 400]}
    elif name == "xgboost":
        pipe = build_classifier(name, random_state)
        grid = {"max_depth": [4, 6], "learning_rate": [0.03, 0.05], "n_estimators": [200, 400]}
    elif name == "gbdt":
        pipe = build_classifier(name, random_state)
        grid = {"max_depth": [3, 4], "learning_rate": [0.03, 0.05], "n_estimators": [200, 400]}
    elif name == "rf":
        pipe = build_classifier(name, random_state)
        grid = {"max_depth": [8, 12], "n_estimators": [200, 400]}
    else:
        raise ValueError(name)

    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    search.fit(x, y)
    return search.best_estimator_


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
    return model.predict(x).astype(float)
