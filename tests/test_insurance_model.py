import numpy as np
import pandas as pd
import pytest

from insurance_model import load_dataset, predict_charges, train_and_evaluate


def sample_data(rows=40):
    rng = np.random.default_rng(42)
    age = rng.integers(18, 65, rows)
    bmi = rng.uniform(18, 42, rows)
    children = rng.integers(0, 5, rows)
    smoker = rng.choice(["yes", "no"], rows)
    sex = rng.choice(["male", "female"], rows)
    region = rng.choice(["northwest", "northeast", "southwest", "southeast"], rows)
    charges = 500 + age * 180 + bmi * 90 + children * 250 + (smoker == "yes") * 18000
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges.astype(float),
        }
    )


def test_load_dataset_validates_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_checks_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [20], "charges": [100]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_training_selects_model_and_predicts_unseen_category():
    frame = sample_data()
    model, leaderboard, metrics = train_and_evaluate(frame, cv_splits=3)
    assert len(leaderboard) == 3
    assert leaderboard["cv_rmse_mean"].is_monotonic_increasing
    assert metrics["test_rows"] > 0
    assert metrics["rmse"] >= 0

    prediction = predict_charges(
        model,
        {
            "age": 35,
            "sex": "female",
            "bmi": 27.5,
            "children": 1,
            "smoker": "no",
            "region": "unknown-region",
        },
    )
    assert np.isfinite(prediction)
    assert prediction >= 0


def test_prediction_rejects_missing_features():
    frame = sample_data()
    model, _, _ = train_and_evaluate(frame, cv_splits=3)
    with pytest.raises(ValueError, match="Missing applicant fields"):
        predict_charges(model, {"age": 30})
