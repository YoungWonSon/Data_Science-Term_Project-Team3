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

# 3. Preprocessing & Feature Engineering Lead (Cho Minseo)
1. Replace missing value with ffill
- The dataset we use has data values over continuous time, so replace using the ffill function.
2. Data type preprocessing
- To apply the modeling, classify text data and numerical data, and convert the values of the 'Date' column into integers.
3. Remove error
- Eliminating Error Outliers (-99.0, 999.0).
4. Find outlier using IQR method
- Using IQR method to deal with outlier, but for our data analysis goal, we believe that outlier should be maintained without getting rid of it.
5. Apply robust scaling considering outlier
- Reduce the influence of extreme values with robust scaling instead of handling outliers.
6. Create a KDE graph before and after scaling
- Visualize before and after scaling at a glance.
7. Apply one-hot-encoding
- Apply encoding to use the object type 'Country' feautre in the algorithm, and use the one hot encoding method because the country is not ranked or ordered.
---

# 4. Classification, Clustering, Regression Modeling & Evaluation Lead

k-Means Clustering vs Hierarchical Agglomerative Clustering
1. 숫자형 변수는 표준화, Country는 one-hot encoding = 이런 구조는 K-Means가 centroid를 기준으로 패턴을 나누기에 적합한 형태
2. K-Means: 클러스터 중심점인 centroid를 기준으로 데이터를 반복적으로 분류.
            => 데이터 개수가 많아도 비교적 빠르게 실행
   Hierarchical Agglomerative Clustering: 처음에 각 데이터를 하나의 클러스터로 보고, 가장 가까운 클러스터끼리 계속 병합
                                       => 전체 거리 관계를 많이 계산해야 하기 때문에 데이터가 많아질수록 느림
3. K-Means: 각 클러스터의 중심값 확인 가능 => K-Means는 클러스터별 특징을 설명하기 쉬움
   Hierarchical Clustering: dendrogram을 통해 데이터가 어떻게 병합되는지는 볼 수 있지만, 
                            최종 클러스터의 특징을 직관적으로 해석하려면 추가 분석이 필요
                            
inertia = K-Means가 만든 클러스터가 얼마나 잘 모여 있는지를 나타내는 값
          각 데이터가 자기 클러스터의 중심점에서 얼마나 떨어져 있는지의 총합
inertia가 작다 = 데이터들이 중심점 주변에 잘 모여 있다
inertia가 크다 = 데이터들이 중심점에서 많이 흩어져 있다

Elbow Method: 여러 개의 k 값을 실험해 보고, 각 k의 inertia를 그래프로 그린 뒤 적절한 클러스터 개수를 고르는 방법
              클러스터 수를 늘렸을 때 성능 향상이 갑자기 줄어드는 지점을 찾는 방법

Silhouette Score: 클러스터링이 얼마나 잘 되었는지를 평가하는 점수
                  1) Cluster 안에서 잘 뭉쳐 있는가?
                  2) 다른 Cluster와 잘 구분되는가?
                  => 내 클러스터 안에서는 가깝고, 다른 클러스터와는 멀어야 좋다.
                1에 가까움 = 클러스터가 매우 잘 나뉨

Decision Tree vs KNN
1. 경제 지표들이 여러 개 있는 경우에는 Decision Tree가, 어떤 조건 때문에 유가 상승이 크다고 판단했는지 설명하기 좋음
2. 수치형 경제 변수 많음 = (KNN에서의 문제)차원이 많아질수록 거리 계산이 덜 명확해짐, 때문에 Descision Tree 선택
3. 어떤 변수가 유가 상승 분류에 중요한지 feature importance로 설명할 수 있다.

1. 여러 독립변수 X가 존재 => 따라서 하나의 변수만 사용하는 simple linear regression보다, 
                            여러 변수를 동시에 사용하는 multiple linear regression이 더 적합
2. Polynomial Regression은 비선형 관계를 잡을 수 있다는 장점이 있음, 
   but 변수가 많은 상태에서 polynomial regression을 적용하면 문제가 생길 수 있음. 
   특히 2차 polynomial 적용 = 기존 변수 + 변수의 제곱 + 변수끼리의 곱 
        => 2차 polynomial을 적용하면 feature 수가 급격히 늘어남, 연산량 상승
        
Mean Squared Error, MSE = 평균((실제값 - 예측값)²), 모델이 얼마나 틀렸는지
Root Mean Squared Error, RMSE = RMSE는 MSE에 루트를 씌운 값
R-squared, R² Score = 1 - (모델의 오차 / 평균 모델의 오차), 모델이 target의 변화를 얼마나 잘 설명하는지
                      전체 데이터 변동 중에서 모델이 설명할 수 있는 비율
                      R² = 0.85이면, 모델이 Fuel_Price_Local의 변동을 약 85% 설명한다.
| R² 값      | 의미                     |
| 1에 가까움 | 모델이 데이터를 매우 잘 설명함      |
| 0에 가까움 | 모델이 평균값으로 예측하는 것과 비슷함  |
| 음수       | 모델이 평균값으로 예측하는 것보다도 못함 |

Regression Coefficients = 각 독립변수가 target에 얼마나 영향을 주는지를 나타내는 값
                         if WTI_Crude_USD_per_barrel의 Regression Coefficients가 0.45이면 
                            => WTI 원유 가격이 증가할수록 Fuel_Price_Local도 증가하는 경향이 있다.
                         if News_Sentiment_Score = -0.20 => 뉴스 감성 점수가 높아질수록 Fuel_Price_Local은 감소하는 경향이 있다.
| Coefficient 부호 | 의미                          |
| 양수 `+`         | 해당 변수가 증가하면 target도 증가하는 경향 |
| 음수 `-`         | 해당 변수가 증가하면 target은 감소하는 경향 |
| 0에 가까움       | target에 미치는 영향이 약함          |


| 지표                     | 의미                             | 좋은 방향       |
| MSE                     | 실제값과 예측값 차이의 제곱 평균   | 작을수록 좋음     |
| RMSE                    | MSE의 제곱근, 평균적인 예측 오차   | 작을수록 좋음     |
| R² Score                | 모델이 target 변동을 설명하는 비율 | 1에 가까울수록 좋음 |
| Regression Coefficients | 각 독립변수가 target에 미치는 영향 | 부호와 크기로 해석  |

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
Kwon_Keonho_Regression_OpenSource.py
Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv
```

Run the script:

```bash
python Kwon_Keonho_Regression_OpenSource.py
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
