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

# 5. Regression Modeling & Open Source SW Lead

**Author:** Kwon Keonho  
**Main Code:** `Kwon_Keonho_Regression_OpenSource.py`  
**Required Dataset:** `Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv`

---

## 5.1 Role Overview

This module was developed for the **Regression Modeling & Open Source SW Lead** role in Team 3's Data Science Term Project.

The main purpose of this part is to predict the **concrete next-day local fuel price change amount** and to organize the regression process as a reusable open-source style pipeline.

This code integrates:

- data cleaning
- missing value handling
- categorical encoding
- feature scaling
- regression target generation
- model training and testing
- cross-validation
- model combination comparison
- evaluation metric calculation
- output file generation

into one top-level function.

---

## 5.2 Prediction Target

The regression target variable is:

```text
Next_Day_Fuel_Change_Amount
= next day's Fuel_Price_Local - today's Fuel_Price_Local
```

This means the model predicts **how much the local fuel price will change on the next day**, rather than only predicting whether the price will rise or fall.

---

## 5.3 Main Top-Level Function

The full regression workflow is executed by one function:

```python
run_fuel_price_regression_pipeline()
```

### Function Parameters

| Parameter | Default | Description |
|---|---:|---|
| `csv_path` | `Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv` | Path to the dirty dataset CSV file |
| `test_size` | `0.2` | Ratio of test data |
| `cv` | `5` | Number of folds for cross-validation |
| `random_state` | `42` | Random seed for reproducibility |
| `output_dir` | `regression_outputs` | Folder for saved result files |
| `save_outputs` | `True` | Saves CSV results and plot image when True |
| `include_polynomial` | `False` | Also compares Polynomial Regression Degree 2 when True |

---

## 5.4 Internal Team Code References

This code reuses and extends several team members' code ideas. The source of each reused or adapted part is clearly documented inside the Python code.

| Used Part | Internal Reference Code | How It Was Used |
|---|---|---|
| EDA and dirty value detection | `Seo_Jangwon_EDA.py` | Used as a reference for checking dataset structure, missing values, and abnormal values such as `999.0` and `-99.0` |
| Missing value handling | `Jo_Minseo_fillna.py` | Used as a reference for country-level time-series missing value handling |
| One-hot encoding | `Jo_Minseo_Onehotencoding.py` | Used as a reference for applying one-hot encoding to the `Country` column |
| Scaling strategy | `Jo_Minseo_Scaling.py`, `Kim_Dana_Standarization_Data.py` | Used as references for StandardScaler, RobustScaler, and standardized data construction |
| Outlier handling | `Jo_Minseo_Outlier.py`, `Jo_Minseo_Outlierdelete.py` | Used as references for detecting and handling error-type outliers |
| Baseline regression | `KIm_Dana_Multi-Linear-Regression.py` | Extended from the baseline multiple linear regression structure |

This module does not simply duplicate the baseline regression code. It extends the regression task by redefining the target variable as `Next_Day_Fuel_Change_Amount` and combining preprocessing, model comparison, evaluation, and output saving into a single reusable pipeline.

---

## 5.5 Preprocessing Workflow

### 1. Dirty Value Cleaning

The following values are treated as obvious error-type dirty values:

```text
Fuel_Price_Change_Percent == 999.0
News_Sentiment_Score == -99.0
```

These values are converted to `NaN` and later handled during preprocessing. Extreme values are also clipped to reduce excessive influence on the regression model.

### 2. Missing Value Handling

Missing numeric time-series values are filled by country using the following order:

```text
country-level linear interpolation
→ forward fill / backward fill
→ global median replacement if needed
```

### 3. Categorical Encoding

The `Country` column is encoded using one-hot encoding because country names do not have ordinal meaning.

### 4. Feature Scaling

The pipeline compares two scaling methods:

- `StandardScaler`
- `RobustScaler`

This helps reduce scale differences among crude oil prices, exchange rates, local fuel prices, and other numerical indicators.

---

## 5.6 Regression Models

The main model used in this module is:

```text
Multiple Linear Regression
```

The code also supports:

```text
Polynomial Regression Degree 2
```

Polynomial regression is optional and is disabled by default because it creates many additional interaction features.

---

## 5.7 Model Combination Comparison

The pipeline compares different combinations of:

- feature sets
- missing value strategies
- scaling methods
- regression degree

### Compared Feature Sets

| Feature Set | Description |
|---|---|
| `all_features` | Uses all available features |
| `without_current_change_percent` | Removes `Fuel_Price_Change_Percent` |
| `without_Brent_for_multicollinearity` | Removes `Brent_Crude_USD_per_barrel` |
| `without_WTI_for_multicollinearity` | Removes `WTI_Crude_USD_per_barrel` |

The best model is selected mainly by the lowest `CV_RMSE` and then by the lowest `Test_RMSE`.

---

## 5.8 Evaluation Metrics

The regression model is evaluated using:

| Metric | Meaning | Better Direction |
|---|---|---|
| MAE | Mean Absolute Error | Lower is better |
| MSE | Mean Squared Error | Lower is better |
| RMSE | Root Mean Squared Error | Lower is better |
| R² | Coefficient of Determination | Closer to 1 is better |

MAE was added to make the average prediction error easier to interpret, while MSE, RMSE, and R² were extended from the baseline regression evaluation structure.

---

## 5.9 Output Files

When the script is executed, the following files are saved in:

```text
regression_outputs/
```

| Output File | Description |
|---|---|
| `all_regression_model_results.csv` | Evaluation results of all model combinations |
| `top5_regression_model_results.csv` | Top 5 model combinations |
| `regression_predictions.csv` | Actual values, predicted values, and residuals |
| `top30_regression_coefficients.csv` | Top 30 regression coefficients by absolute value |
| `actual_vs_predicted.png` | Actual vs predicted scatter plot |

---

## 5.10 How to Run

Place the code file and dataset file in the same directory:

```text
Kwon_Keonho_Regression_OpenSource_with_Team_References.py
Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv
```

Run the script:

```bash
python Kwon_Keonho_Regression_OpenSource_with_Team_References.py
```

After execution, result files will be generated in:

```text
regression_outputs/
```

---

## 5.11 Required Libraries

```text
pandas
numpy
matplotlib
scikit-learn
```

Main scikit-learn components used:

```text
ColumnTransformer
SimpleImputer
LinearRegression
KFold
cross_validate
train_test_split
Pipeline
OneHotEncoder
PolynomialFeatures
RobustScaler
StandardScaler
mean_absolute_error
mean_squared_error
r2_score
```

---

## 5.12 Summary

This module is the final regression and open-source software contribution part of the project. It uses internal team code references for preprocessing and baseline modeling, then extends them into a unified regression pipeline.

The key contribution is the creation of a reusable top-level function that predicts `Next_Day_Fuel_Change_Amount`, compares multiple model combinations, evaluates performance with MAE, MSE, RMSE, and R², and saves result files for the final report and presentation.
