'''
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

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Kim_Dana_Standarization_Data import df_standardized

# 1. 데이터 확인
print("Data Shape:", df_standardized.shape)
print(df_standardized.head())


# 2. Target column 설정, 예측하고 싶은 컬럼(다른 것으로 얼마든지 대체 가능)
target_col = "Fuel_Price_Local"

# 3. 필요하지 않은 컬럼 제거, Date는 문자열/날짜형이므로 Linear Regression에 바로 넣기 어려움
# target_col은 y로 따로 분리해야 하므로 X에서 제거
X = df_standardized.drop(columns=["Date", target_col])
y = df_standardized[target_col]

# NaN 제거 후 다시 X, y 분리
model_df = pd.concat([X, y], axis=1)
model_df = model_df.dropna()
X = model_df.drop(columns=[target_col])
y = model_df[target_col]


# 4. Train / Test Split
# 데이터를 학습용 80%, 테스트용 20%로 나눔
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. Multiple Linear Regression 모델 생성 및 학습
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# 6. 예측
y_pred = mlr_model.predict(X_test)

# 7. 모델 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n[Multiple Linear Regression Results]")
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R-squared (R2 Score):", round(r2, 4))


# 8. 회귀 계수 확인
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": mlr_model.coef_
})
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

print("\n[Regression Coefficients]")
print(coefficients)


# 9. 실제값 vs 예측값 비교
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print("\n[Actual vs Predicted]")
print(comparison.head(10))

# 10. 시각화
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Fuel_Price_Local")
plt.ylabel("Predicted Fuel_Price_Local")
plt.grid(True)
plt.show()