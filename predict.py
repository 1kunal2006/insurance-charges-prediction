"""Predict insurance charges using a previously trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from insurance_model import predict_charges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="artifacts/insurance_model.joblib",
        help="Path to a model created by train_model.py",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON object with age, sex, bmi, children, smoker, and region",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        parser.error(f"Model not found at '{model_path}'. Run train_model.py first.")

    try:
        applicant = json.loads(args.input)
        if not isinstance(applicant, dict):
            raise ValueError("Input must be a JSON object.")
        prediction = predict_charges(joblib.load(model_path), applicant)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(json.dumps({"predicted_charges": round(prediction, 2)}, indent=2))


if __name__ == "__main__":
    main()
