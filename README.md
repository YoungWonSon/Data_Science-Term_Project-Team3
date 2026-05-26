# Data_Science-Term_Project-Team3
To analyze the impact of global external shocks on domestic oil prices, building a predictive model to help students commuting by car and transport workers proactively respond to oil price fluctuations.


# 2.Performed Exploratory Data Analysis (EDA) for the global fuel crisis dataset.(seojangwon)
1.Checked dataset structure and column information
2.Analyzed missing values
3.Investigated abnormal error values such as 999.0 and -99.0
4.Visualized data distribution using histograms
5.Analyzed outliers using boxplots
6.Analyzed correlations between variables using heatmap
7.Compared fuel price distributions by country
# explanation
Oil price related variables showed strong positive correlations, while stock market index changes showed negative correlations with crisis-related variables.
Some columns contained missing values and error-type values (999.0, -99.0), and large fluctuations were preserved because they may represent real crisis situations.


# 5. Regression Modeling & Open Source SW Lead – Kwon Keonho

## Role Summary

This section describes the work completed by **Kwon Keonho** as the **Regression Modeling & Open Source SW Lead** in Team 3's Data Science Term Project.

The main responsibility of this role was to build a regression-based prediction module that estimates the **concrete next-day fuel price change amount** and to organize the final code in an open-source style structure.

## Module Objective

The regression module predicts how much the local fuel price will change on the next day.

The target variable is defined as:

```text
Next_Day_Fuel_Change_Amount
= next day's Fuel_Price_Local - today's Fuel_Price_Local
```

This means the model does not only predict whether the fuel price increases or decreases.  
It predicts the actual amount of price movement.

## Main Code File

```text
Kwon_Keonho_Regression_OpenSource.py
```

This file contains the final regression pipeline for the project.

## Main Top-Level Function

```python
run_fuel_price_regression_pipeline()
```

This function runs the full regression process in one workflow:

```text
Data loading
→ Dirty value cleaning
→ Missing value handling
→ Target variable generation
→ Encoding and scaling
→ Regression model training
→ Cross-validation
→ Final evaluation
→ Output saving
```

## Preprocessing Included

The module includes the following preprocessing steps:

- Converts obvious dirty values such as `999.0` and `-99.0` into missing values
- Handles missing values by country-level time-series interpolation and fill methods
- Applies one-hot encoding to the `Country` column
- Applies feature scaling using `StandardScaler` or `RobustScaler`
- Creates the regression target variable from country-level next-day fuel price changes

## Regression Model

The main model used in this module is:

```text
Multiple Linear Regression
```

The module also supports optional polynomial regression, but it is turned off by default to keep the final version simple and efficient.

## Model Comparison

The code compares multiple combinations of:

- Feature sets
- Missing value strategies
- Scaling methods
- Regression model settings

The best model is selected based mainly on cross-validation RMSE and final test RMSE.

## Evaluation Metrics

The regression model is evaluated with:

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

These metrics are used to measure how accurately the model predicts the next-day fuel price change amount.

## Output Files

After running the code, the following files are generated in the `regression_outputs/` folder:

| File | Description |
|---|---|
| `all_regression_model_results.csv` | Full results of all tested model combinations |
| `top5_regression_model_results.csv` | Top 5 model combinations |
| `regression_predictions.csv` | Actual values, predicted values, and residuals |
| `top30_regression_coefficients.csv` | Major regression coefficients |
| `actual_vs_predicted.png` | Actual vs predicted visualization |

## How to Run

Place the following files in the same folder:

```text
Kwon_Keonho_Regression_OpenSource.py
Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv
```

Then run:

```bash
python Kwon_Keonho_Regression_OpenSource.py
```

## Required Libraries

```text
pandas
numpy
matplotlib
scikit-learn
```

## Contribution Summary

This module contributes to the project by:

1. Defining a regression target for next-day fuel price movement
2. Building a reusable top-level regression pipeline
3. Comparing multiple preprocessing and model combinations
4. Evaluating model performance with MAE, MSE, RMSE, and R²
5. Saving outputs that can be used in the final report, presentation, and GitHub repository

In summary, this part connects the project’s preprocessing results with the final regression modeling task and provides an open-source style structure suitable for submission.
