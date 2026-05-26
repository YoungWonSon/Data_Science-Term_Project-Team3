import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 역할: Data Description & EDA
# 목적: 데이터 구조, 결측치, 이상치, 분포, 상관관계를 확인하여
# 이후 전처리와 모델링이 필요한 이유를 분석한다.

# 0. 데이터셋 불러오기
df = pd.read_csv("/Users/seojang-won/Downloads/Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")


# 1. 데이터 기본 구조 확인
print("Dataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nColumns:")
print(df.columns)


# 2. 기초 통계량 분석
print("\nStatistical Summary:")
print(df.describe())


# 3. 결측치 분석
# 각 컬럼별 결측치 개수를 확인한다.
# 결측치가 있는 컬럼은 이후 전처리 단계에서 보간 또는 대체가 필요하다.
print("\nMissing Values:")
print(df.isnull().sum())


# 4. 오류성 이상치 분석
# 전쟁/위기 상황에서는 큰 변동값이 의미 있는 데이터일 수 있으므로
# IQR 방식으로 모든 이상치를 제거하지 않는다.
# 대신 999.0, -99.0처럼 전산 오류성 값으로 보이는 데이터만 확인한다.

print("\n999.0 Error Values:")
print((df == 999.0).sum())

print("\n-99.0 Error Values:")
print((df == -99.0).sum())


# 5. 히스토그램 시각화
# 전체 수치형 변수들의 분포를 확인한다.
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()


# 6. Fuel_Price_Change_Percent 박스플롯
# 999.0은 전산 오류성 값으로 판단하여 시각화에서 제외한다.
# 일반적인 큰 변동값은 전쟁/위기 상황에서 의미 있을 수 있으므로 유지한다.
filtered_data = df[df["Fuel_Price_Change_Percent"] < 100]

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=filtered_data["Fuel_Price_Change_Percent"],
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "red",
        "markersize": 4
    }
)

plt.title("Fuel Price Change Percent Boxplot", fontsize=15)
plt.xlabel("Fuel Price Change Percent")
plt.tight_layout()
plt.show()


# 7. 상관관계 히트맵
# 수치형 변수들 사이의 상관관계를 확인한다.
plt.figure(figsize=(13, 9))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=9)
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()


# 8. 국가별 연료 가격 분포 박스플롯
# 국가별 Fuel_Price_Local 분포 차이를 확인한다.
# 값 차이가 큰 국가들이 있어 로그 스케일을 적용한다.
plt.figure(figsize=(14, 7))

sns.boxplot(
    x="Country",
    y="Fuel_Price_Local",
    data=df,
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "red",
        "markersize": 4
    }
)

plt.yscale("log")
plt.title("Fuel Price Distribution by Country", fontsize=16)
plt.xlabel("Country")
plt.ylabel("Fuel Price Local (Log Scale)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 9. 상관관계 높은 변수 조합 확인
# 히트맵에서 숫자가 많아 보기 어려우므로,
# 상관관계가 높은 변수 조합을 따로 정리한다.

corr_matrix = df.corr(numeric_only=True).abs()

corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# 자기 자신과의 상관관계 1.0은 제외
corr_pairs = corr_pairs[corr_pairs < 1]

print("\nTop 10 Highly Correlated Feature Pairs:")
print(corr_pairs.head(10))
