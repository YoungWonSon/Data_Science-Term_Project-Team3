import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from Kim_Dana_Standarization_Data import df_standardized

# 1. 데이터 확인
print("Data Shape:", df_standardized.shape)
print(df_standardized.head())


# 2. Target column 설정, 예측하고 싶은 컬럼(다른 것으로 얼마든지 대체 가능)
target_col = "Fuel_Price_Change_Percent"

# 3. 필요하지 않은 컬럼 제거, Date는 문자열/날짜형이므로 Linear Regression에 바로 넣기 어려움
# target_col은 y로 따로 분리해야 하므로 X에서 제거
X = df_standardized.drop(columns=["Date", target_col])

country_cols = [col for col in X.columns if col.startswith("Country_")]
y = df_standardized[target_col]
X = X.drop(columns=country_cols)

model_df = pd.concat([X, y], axis=1)
model_df = model_df.dropna()

X = model_df.drop(columns=[target_col])
y = model_df[target_col]
print(model_df.head())

# 4. Train / Test Split
# 데이터를 학습용 80%, 테스트용 20%로 나눔
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    random_state=42
)
'''
# 5. Multiple Linear Regression 모델 생성 및 학습
mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)
'''
mlr_model = Ridge(alpha=1.0)
mlr_model.fit(X_train, y_train)

# 6. 예측
y_pred = mlr_model.predict(X_test)

# 7. 모델 평가
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
y_true = y_test.values
y_pred_array = y_pred

print("\n[Multiple Linear Regression Results]")
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R-squared (R2 Score):", round(r2, 4))

# SST: Total Sum of Squares, 실제값이 평균으로부터 얼마나 떨어져 있는지
sst = np.sum((y_true - np.mean(y_true)) ** 2)

# SSE: Sum of Squared Errors, 실제값과 예측값의 오차 제곱합
sse = np.sum((y_true - y_pred_array) ** 2)

# SSR: Regression Sum of Squares, 예측값이 실제값 평균으로부터 얼마나 떨어져 있는지
ssr = np.sum((y_pred_array - np.mean(y_true)) ** 2)

'''
# MAPE: Mean Absolute Percentage Error, y_true가 0이면 나눗셈 오류가 나므로 0이 아닌 값만 사용
nonzero_mask = y_true != 0
mape = np.mean(
    np.abs((y_true[nonzero_mask] - y_pred_array[nonzero_mask]) / y_true[nonzero_mask])
) * 100
'''
print("\n[Regression Sum of Squares]")
print("SST:", round(sst, 4))
print("SSE:", round(sse, 4))
print("SSR:", round(ssr, 4))
# print("MAPE (%):", round(mape, 4))

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

plt.plot([min_value, max_value], [min_value, max_value], 
         linestyle="--", label="Perfect Prediction Line")

trend_coef = np.polyfit(y_test, y_pred, 1)
trend_line = np.poly1d(trend_coef)
x_line = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(
    x_line,
    trend_line(x_line),
    linewidth=2,
    label="Prediction Trend Line"
)

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Fuel_Price_Change_Percent")
plt.ylabel("Predicted Fuel_Price_Change_Percent")
plt.grid(True)
plt.show()