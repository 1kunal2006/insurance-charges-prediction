# Insurance Charges Prediction

**Author: Kunal**

## Project Overview

This project explores a medical insurance dataset to understand the factors associated with insurance charges. The notebook focuses on analyzing the data, preparing it for further use, and identifying relationships between different features and the target variable, `charges`.

## Project Objectives

- Explore the insurance dataset and understand its main characteristics.
- Identify patterns and relationships between personal attributes and medical insurance charges.
- Clean the data and check for missing values and duplicate records.
- Prepare categorical and numerical features for analysis.
- Create BMI categories as an example of feature engineering.
- Explore feature relationships and select relevant features for future predictive modeling.

## Dataset Description

The dataset includes the following attributes:

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `sex` | Sex of the individual |
| `bmi` | Body Mass Index |
| `children` | Number of children or dependents covered |
| `smoker` | Whether the individual smokes |
| `region` | Residential region |
| `charges` | Medical insurance charges (target variable) |

Obtain the dataset from the source permitted for your use. If it is not included in this repository, upload the CSV to your notebook environment or update the file path in the notebook.

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
Review the dataset structure, summary statistics, feature distributions, boxplots, and correlations to understand the data and identify possible patterns.

### 2. Data Cleaning
Inspect the data for missing values and duplicate records, and remove duplicate rows where appropriate.

### 3. Data Preprocessing
Prepare categorical and numerical columns for analysis using encoding and scaling where appropriate.

### 4. Feature Engineering
Create BMI categories from the BMI feature to explore whether grouping BMI values helps reveal patterns in insurance charges.

### 5. Feature Selection
Examine relationships between features using techniques such as Pearson correlation and chi-square tests, where appropriate for the feature types.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- SciPy

## Repository Contents

- `insurance.ipynb` — Jupyter/Google Colab notebook containing the project's exploratory analysis, preprocessing, feature engineering, and feature-selection work.

## How to Run the Project

1. Clone or download this repository.
2. Open `insurance.ipynb` in Google Colab or Jupyter Notebook.
3. Obtain the dataset and upload it to your notebook environment, or update the dataset path in the relevant notebook cell.
4. Run the notebook cells in order.
5. Review the visualizations and analysis outputs to understand the dataset and the factors associated with insurance charges.

## Current Scope

The project focuses on understanding and preparing the insurance data. It is intended to demonstrate the data-analysis workflow and provide a foundation for future predictive modeling; it does not claim to provide validated insurance-cost predictions.

## Future Improvements

- Train regression models to predict insurance charges.
- Compare models using appropriate regression metrics.
- Investigate feature importance and prediction errors.
- Improve preprocessing and evaluate model performance on unseen data.

---

**Created by Kunal**
