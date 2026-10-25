"""Reusable preprocessing, model selection, and prediction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]
RANDOM_STATE = 42


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the expected insurance CSV dataset."""
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Download the insurance CSV "
            "and pass its path with --data."
        )

    frame = pd.read_csv(dataset_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = frame[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # A missing target cannot be used for supervised training; feature missing
    # values are imputed inside the training pipeline to avoid data leakage.
    frame = frame.dropna(subset=[TARGET])
    if frame.empty:
        raise ValueError("Dataset has no rows with a valid 'charges' target.")
    if (frame[TARGET] < 0).any():
        raise ValueError("The 'charges' target must be non-negative.")

    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").str.strip().str.lower()
        frame[column] = frame[column].replace("", pd.NA).astype(object)

    frame = frame.drop_duplicates().reset_index(drop=True)
    if len(frame) < 10:
        raise ValueError("At least 10 valid rows are required to train and evaluate.")
    return frame


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fitted only on each training fold."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_candidates() -> dict[str, Any]:
    """Return deterministic baseline models suitable for tabular regression."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    cv_splits: int = 5,
) -> tuple[Pipeline, pd.DataFrame, dict[str, float]]:
    """Select a model by training-only cross-validation, then evaluate once."""
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")
    if len(frame) < max(10, cv_splits * 2):
        raise ValueError(f"At least {max(10, cv_splits * 2)} rows are needed.")

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing)}")

    X = frame[FEATURES].copy()
    y = pd.to_numeric(frame[TARGET], errors="raise")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    candidates = []
    for name, estimator in build_model_candidates().items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            n_jobs=1,
        )
        candidates.append(
            {
                "model": name,
                "cv_rmse_mean": float(-scores.mean()),
                "cv_rmse_std": float(scores.std()),
                "estimator": estimator,
            }
        )

    candidates.sort(key=lambda item: item["cv_rmse_mean"])
    best = candidates[0]
    selected_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", best["estimator"]),
        ]
    )
    selected_pipeline.fit(X_train, y_train)
    predictions = selected_pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
    }
    leaderboard = pd.DataFrame(
        [
            {
                "model": item["model"],
                "cv_rmse_mean": item["cv_rmse_mean"],
                "cv_rmse_std": item["cv_rmse_std"],
            }
            for item in candidates
        ]
    )
    return selected_pipeline, leaderboard, metrics


def predict_charges(model: Pipeline, applicant: dict[str, Any]) -> float:
    """Predict charges for one applicant after validating required features."""
    missing = sorted(set(FEATURES) - set(applicant))
    if missing:
        raise ValueError(f"Missing applicant fields: {', '.join(missing)}")

    row = pd.DataFrame([{column: applicant[column] for column in FEATURES}])
    for column in NUMERIC_FEATURES:
        row[column] = pd.to_numeric(row[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        row[column] = row[column].astype(str).str.strip().str.lower()

    prediction = float(model.predict(row)[0])
    return max(0.0, prediction)


def save_model(model: Pipeline, path: str | Path) -> Path:
    """Persist the complete fitted pipeline, including preprocessing."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path
