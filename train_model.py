"""Train and compare insurance charge regression models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insurance_model import load_dataset, save_model, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="insurance.csv", help="Path to insurance CSV")
    parser.add_argument(
        "--model-out",
        default="artifacts/insurance_model.joblib",
        help="Where to save the fitted model pipeline",
    )
    parser.add_argument(
        "--metrics-out",
        default="artifacts/metrics.json",
        help="Where to save evaluation metrics",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="Cross-validation folds")
    args = parser.parse_args()

    if args.cv_splits < 2:
        parser.error("--cv-splits must be at least 2")

    data = load_dataset(args.data)
    model, leaderboard, metrics = train_and_evaluate(data, cv_splits=args.cv_splits)
    model_path = save_model(model, args.model_out)

    metrics_path = Path(args.metrics_out).expanduser()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "selected_model": str(model.named_steps["model"]),
                "metrics": metrics,
                "cross_validation": leaderboard.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nCross-validation leaderboard (lower RMSE is better):")
    print(leaderboard.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    print("\nHeld-out test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
