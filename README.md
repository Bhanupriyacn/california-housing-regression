# California Housing Regression: Linear, Ridge, and Lasso

Day 2 Machine Learning Fundamentals assignment using the Kaggle California Housing Prices dataset.

## Problem Statement

Build a Linear Regression model to predict a continuous target variable. Then apply Ridge and Lasso regularization and compare coefficients. Evaluate all three models using MAE, MSE, RMSE, and R2, and state which model generalizes best and why.

## Dataset

- Source: https://www.kaggle.com/datasets/camnugent/california-housing-prices
- File: `housing.csv`
- Target: `median_house_value`

## Deliverables

- Executed notebook: `california_housing_linear_ridge_lasso.ipynb`
- Train/test split with shared preprocessing pipeline
- Linear Regression, RidgeCV, and LassoCV models
- Metrics comparison table with MAE, MSE, RMSE, and R2
- Coefficient comparison across all three models
- Short conclusion on best model and regularization effect

## Result Summary

Linear Regression was slightly best on the held-out test set, with RMSE about 70,059 and R2 about 0.6254. Ridge and Lasso performed almost identically, so regularization did not materially improve test accuracy for this feature set, though it did shrink/simplify coefficients.
