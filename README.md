# Insurance Charges Prediction

An end-to-end machine-learning project for exploring and predicting medical insurance charges from demographic and lifestyle features. The original exploratory notebook is retained; a reproducible training pipeline and prediction CLI are provided alongside it.

## Project contents

- `insurance.ipynb` — existing exploratory data analysis and regression notebook (originally created in Google Colab).
- `insurance_model.py` — validated dataset loading, leakage-safe preprocessing, model candidates, evaluation, and prediction helpers.
- `train_model.py` — trains and compares Linear Regression, Ridge Regression, and Random Forest models using cross-validation on the training split; evaluates the selected model on a held-out test split.
- `predict.py` — predicts charges from a JSON applicant record using the saved pipeline.
- `tests/` — automated tests for dataset validation, model training, and prediction.
- `requirements.txt` — Python dependencies.

## Dataset

The expected CSV has these columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and `charges`. Use a legally obtained copy of the commonly used medical insurance charges dataset and place it at `insurance.csv` in the project root, or pass another path with `--data`. The dataset is not committed to this repository.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

```bash
python train_model.py --data insurance.csv
```

This writes the fitted preprocessing-plus-model pipeline to `artifacts/insurance_model.joblib` and evaluation details to `artifacts/metrics.json`. The leaderboard reports mean and standard deviation of cross-validated RMSE; lower RMSE is better. Test metrics include MAE, RMSE, and R². Results are computed when you run the script and are not hard-coded into this README.

To use another dataset path or output path:

```bash
python train_model.py --data /path/to/insurance.csv \
  --model-out artifacts/model.joblib \
  --metrics-out artifacts/metrics.json
```

## Predict for an applicant

Train the model first, then provide all six input features as JSON:

```bash
python predict.py --input '{"age": 35, "sex": "female", "bmi": 27.5, "children": 1, "smoker": "no", "region": "southwest"}'
```

The output is a JSON object containing `predicted_charges`. The prediction is an estimate from the public training dataset, not an insurance quote or a guarantee of actual costs.

## Run tests

```bash
pytest -q
```

## Modeling and data-quality notes

- Numeric imputation/scaling and categorical imputation/one-hot encoding are fitted inside an sklearn pipeline, so preprocessing is learned only from each training fold.
- Unknown categorical levels at prediction time are tolerated by one-hot encoding.
- The held-out test split is not used to select the model.
- Duplicate rows are removed and rows with missing target values are excluded. Missing feature values are imputed by the pipeline.
- This small, public dataset is useful for learning and demonstration. It is not sufficient to establish actuarial accuracy, fairness, or suitability for real underwriting decisions.

## Notebook

The notebook currently contains Google Colab-specific data-loading assumptions and saved cell outputs. For reproducible runs, use the CLI workflow above; update the notebook's CSV path or upload the CSV when running it in Colab/Jupyter.

## License

No license is currently specified. Add an appropriate license before granting reuse rights.
