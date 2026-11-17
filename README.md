# Breast Cancer Diagnosis Using Machine Learning

A Python project for exploring and preprocessing the Wisconsin Breast Cancer dataset for machine-learning-based diagnosis.

## Overview

The project analyzes breast tumour measurements and prepares the data for binary classification.

- **Benign (B):** Encoded as `0`
- **Malignant (M):** Encoded as `1`

The dataset contains 569 records and 30 numerical diagnostic features.

## Dataset

Place the Kaggle dataset file named `data.csv` in the project directory.

The original dataset contains an ID column and an empty `Unnamed: 32` column, which are removed during preprocessing.

## Workflow

1. **Import Libraries** – NumPy, Pandas, Matplotlib, Seaborn and Scikit-learn.
2. **Load Data** – Read and inspect the dataset.
3. **Data Cleaning** – Remove unnecessary columns and check missing values and duplicates.
4. **Target Encoding** – Map `M` to `1` and `B` to `0`.
5. **Exploratory Data Analysis** – Analyze skewness, distributions and feature correlations.
6. **Feature Transformation** – Apply `log1p` to features with absolute skewness greater than 1.
7. **Feature Reduction** – Remove highly correlated features using a correlation threshold of 0.9.
8. **Train-Test Split** – Split the dataset into 80% training and 20% testing data using stratification.
9. **Feature Scaling** – Apply StandardScaler to standardize the features.

### Preprocessing Results

- Original records: 569
- Training records: 455
- Testing records: 114
- Features after reduction: 19
- Train-test split: 80:20

## Requirements

Install the required Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
