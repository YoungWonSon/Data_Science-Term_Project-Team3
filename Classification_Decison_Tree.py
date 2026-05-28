import pandas as pd
import numpy as np
import Kim_Dana_Standarization_Data
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV

#유가 상승 정도를 No_Increase / Slight_Increase / Large_Increase 로 분류하는 것

# 표준화된 데이터와 원본 데이터 가져오기
df_standardized = Kim_Dana_Standarization_Data.df_standardized.copy()
raw_df = Kim_Dana_Standarization_Data.df.copy()

# 1. 이상치 처리
# Fuel_Price_Change_Percent에 999 같은 dirty value가 있으므로 NaN으로 처리
raw_df.loc[raw_df["Fuel_Price_Change_Percent"] > 100, "Fuel_Price_Change_Percent"] = np.nan

# 2. Target 생성: 유가 상승 정도 분류
raw_df["Fuel_Rise_Level"] = pd.cut(
    raw_df["Fuel_Price_Change_Percent"],
    bins=[-np.inf, 0, 5, np.inf],
    labels=["No_Increase", "Slight_Increase", "Large_Increase"],
    include_lowest=True
)

# 3. X, y 설정
# Fuel_Price_Change_Percent는 target을 만드는 데 사용했으므로 제거
X = df_standardized.drop(columns=["Date", "Fuel_Price_Change_Percent"])
date_series = pd.to_datetime(df_standardized["Date"], errors="coerce")

X["Date_Ordinal"] = date_series.map(
    lambda x: x.toordinal() if pd.notna(x) else np.nan
)
y = raw_df["Fuel_Rise_Level"]

# 4. y가 NaN인 행 제거
valid_index = y.notna()
X = X.loc[valid_index]
y = y.loc[valid_index]


###################################################메인 코어
# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# 6. 결측치 처리, imputer는 train data에만 fit해야 함
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 7. Decision Tree 모델 + Grid Search
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 20],
    "class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_search.fit(X_train_imputed, y_train)
dt_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

y_pred = dt_model.predict(X_test_imputed)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# K-Fold Cross Validation 조건 충족
cv_scores = cross_val_score(
    dt_model,
    X_train_imputed,
    y_train,
    cv=5
)

# print each cv score (accuracy) and average them
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))

feature_names = X.columns

# Tree Structure를 그림으로 출력
plt.figure(figsize=(40, 20), dpi=150)

plot_tree(
    dt_model,
    feature_names=list(feature_names),
    class_names=[str(cls) for cls in dt_model.classes_],
    filled=True,
    rounded=True,
    fontsize=6,
    max_depth=4,
    impurity=False
)
plt.title("Decision Tree Structure")
plt.show()

'''
# 이 아래부터는 텀 프로젝트 조건을 제외한, extra 결과 시각화 입니다.

# 요소 중요도 확인 및 시각화
# 10. Feature Importance 확인
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": dt_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances:")
print(importance_df)

# 11. Feature Importance 그래프
top10 = importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top10["Feature"], top10["Importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# 12. Feature별 threshold 묶어서 출력
tree = dt_model.tree_
feature_thresholds = {}

for node_id in range(tree.node_count):
    feature_id = tree.feature[node_id]
    threshold = tree.threshold[node_id]

    if feature_id != -2:
        feature_name = feature_names[feature_id]

        if feature_name not in feature_thresholds:
            feature_thresholds[feature_name] = []

        feature_thresholds[feature_name].append(threshold)

print("\nThresholds by Feature:")

for feature, thresholds in feature_thresholds.items():
    threshold_list = [round(t, 4) for t in thresholds]
    print(f"{feature}: {threshold_list}")
'''