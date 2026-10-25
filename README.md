# Insurance Charges Prediction

**Author: Kunal**

## Project Overview

This project uses exploratory data analysis and machine learning to study the factors associated with medical insurance charges. The notebook explores the dataset, cleans and transforms features, engineers BMI categories, examines feature relationships, and trains a Linear Regression model to estimate insurance charges.

## Project Objectives

- Understand the insurance dataset through exploratory data analysis (EDA).
- Check for missing values and duplicate records.
- Encode categorical variables and scale selected numerical features.
- Create BMI categories through feature engineering.
- Explore feature relationships using correlation and chi-square tests.
- Train a regression model and evaluate its performance on unseen test data.

## Dataset Description

The dataset contains the following features:

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `sex` | Sex of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of children/dependents covered |
| `smoker` | Smoking status |
| `region` | Residential region |
| `charges` | Medical insurance charges — the target variable |

The notebook loads `insurance.csv`. The CSV is not included in the repository, so obtain the dataset separately and update the file path to match your environment.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The notebook reviews the dataset shape, data types, summary statistics, missing values, feature distributions, boxplots, and correlations. These steps help reveal patterns and potential relationships between individual attributes and insurance charges.

### 2. Data Cleaning and Preprocessing

- Create a working copy of the original dataset.
- Remove duplicate rows and inspect missing values.
- Encode `sex` and `smoker` as numeric values.
- Apply one-hot encoding to the `region` column.
- Standardize selected numerical features (`age`, `bmi`, and `children`).

### 3. Feature Engineering and Selection

The notebook groups BMI values into categories (underweight, healthy, overweight, and obese), then encodes those categories. Pearson correlation is used to examine numerical feature relationships with charges, while chi-square tests are used to explore relationships between categorical features and binned charges.

### 4. Train-Test Split

The features are separated from the target variable:

- **X:** input features used to predict charges.
- **y:** the target column, `charges`.

The notebook uses scikit-learn's `train_test_split` with `test_size=0.2` and `random_state=42`. This creates an 80% training set and a 20% test set. The model is fitted using the training set and evaluated on the held-out test set.

### 5. Model Creation and Training

The notebook creates a **Linear Regression** model using scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Linear Regression estimates the relationship between the prepared input features and medical insurance charges. Since charges are continuous numerical values, this is a **regression task**, not a classification task.

### 6. Model Evaluation and Score

The notebook calculates the **R² score** on the test predictions and then calculates **adjusted R²**, which accounts for the number of input features relative to the number of test observations.

The saved notebook output shows an **adjusted R² of approximately 0.8203 (82.03%)**. This is a regression goodness-of-fit metric; it should not be described as classification accuracy. The ordinary R² score is also calculated in the notebook, but the displayed final output is the adjusted R² value.

Other useful regression metrics that could be added for a fuller evaluation are:

- **MAE (Mean Absolute Error):** the average absolute difference between predicted and actual charges.
- **RMSE (Root Mean Squared Error):** gives greater weight to larger prediction errors.
- **R²:** indicates how much variation in the target is explained by the model relative to a baseline.

> **Evaluation note:** The notebook currently performs scaling and feature engineering before the train-test split. For a more rigorous estimate of performance on unseen data, fit preprocessing steps only on the training data (for example, using a scikit-learn Pipeline), then apply the fitted transformations to the test data. This helps prevent information leakage.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- SciPy

## Repository Contents

- `insurance.ipynb` — the notebook containing EDA, cleaning, preprocessing, feature engineering, feature selection, train-test splitting, Linear Regression training, and model evaluation.

## How to Run the Project

1. Clone or download this repository.
2. Open `insurance.ipynb` in Google Colab or Jupyter Notebook.
3. Upload `insurance.csv` to your environment, or update the CSV path in the first data-loading cell.
4. Run the notebook cells in order.
5. Review the plots, feature-selection results, and regression evaluation output.

## Conclusion

This project demonstrates a basic end-to-end data-analysis and regression workflow for insurance charges. It explores the dataset, prepares features, trains a Linear Regression model, and evaluates the predictions with R² and adjusted R². The reported score is specific to the notebook's current preprocessing and train-test setup and should not be interpreted as a guarantee of real-world insurance costs.

---

**Created by Kunal**
