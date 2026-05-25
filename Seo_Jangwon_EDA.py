import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터셋 불러오기
# 프로젝트에서 사용하는 원본 CSV 파일을 Pandas DataFrame 형태로 불러온다.
df = pd.read_csv("/Users/seojang-won/Downloads/Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")


# 데이터 기본 구조 확인
# 데이터가 총 몇 개의 행과 열로 구성되어 있는지 확인한다.
print("Dataset Shape:")
print(df.shape)

# 각 컬럼의 데이터 타입, 결측치 여부, non-null 개수를 확인한다.
# 이를 통해 어떤 컬럼이 숫자형이고 어떤 컬럼이 문자형인지 파악할 수 있다.
print("\nDataset Info:")
print(df.info())

# 전체 컬럼명을 확인하여 데이터셋에 어떤 feature들이 있는지 파악한다.
print("\nColumns:")
print(df.columns)


# 기초 통계량 분석
# 수치형 변수들의 평균, 표준편차, 최소값, 최대값, 사분위수를 확인한다.
# 이 결과를 통해 변수들의 범위 차이와 분포 특성을 파악할 수 있다.
print("\nStatistical Summary:")
print(df.describe())


# 결측치 분석
# 각 컬럼별 결측치 개수를 확인한다.
# 결측치가 많은 컬럼은 이후 전처리 단계에서 대체 또는 보간이 필요하다.
print("\nMissing Values:")
print(df.isnull().sum())


# 이상치 분석
# Fuel_Price_Change_Percent 컬럼에 존재하는 비정상 값 999.0의 개수를 확인한다.
# 999.0은 정상적인 연료 가격 변화율로 보기 어렵기 때문에 이상치로 판단한다.
print("\n999.0 Outliers:")
print((df == 999.0).sum())

# News_Sentiment_Score 컬럼에 존재하는 비정상 값 -99.0의 개수를 확인한다.
# 감성 점수에서 -99.0은 일반적인 범위를 벗어난 값이므로 이상치로 판단한다.
print("\n-99.0 Outliers:")
print((df == -99.0).sum())


# 히스토그램 시각화
# 전체 수치형 변수들의 분포를 확인한다.
# 특정 변수에 값이 몰려 있는지, 오른쪽/왼쪽으로 치우친 분포가 있는지 확인할 수 있다.
df.hist(figsize=(15,10))
plt.tight_layout()
plt.show()


# 박스플롯 시각화

# 999 이상치를 제외한 정상 데이터만 선택
filtered_data = df[df["Fuel_Price_Change_Percent"] < 100]

# Fuel_Price_Change_Percent 이상치 확인
plt.figure(figsize=(8,5))

sns.boxplot(
    x=filtered_data["Fuel_Price_Change_Percent"]
)

plt.title("Fuel Price Change Percent Boxplot")
plt.xlabel("Fuel Price Change Percent")

plt.tight_layout()
plt.show()


# 상관관계 히트맵
# 수치형 변수들 사이의 상관관계를 확인한다.
# 1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계를 의미한다.
plt.figure(figsize=(13,9))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

# x축과 y축 글씨가 겹치지 않도록 회전과 글씨 크기를 조정한다(히트맵 화면이 잘려서 미세조정)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=9)

plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()


# 결측치 히트맵 시각화

plt.figure(figsize=(14,6))

sns.heatmap(
    df.isnull(),
    yticklabels=False,   # y축 숫자 제거
    cbar=False,
    cmap="viridis"
)

plt.title("Missing Value Heatmap", fontsize=16)
plt.xlabel("Columns")
plt.ylabel("Rows")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# 국가별 평균 현지 연료 가격 분석

# Country 컬럼을 기준으로 데이터를 그룹화한 뒤,
# 각 국가별 Fuel_Price_Local의 평균값을 계산한다.
# 이를 통해 국가별 연료 가격 수준 차이를 비교할 수 있다.
country_avg = df.groupby("Country")["Fuel_Price_Local"].mean()

# 국가별 평균 연료 가격을 막대그래프로 시각화한다.
# 국가별 가격 차이가 크기 때문에 한눈에 데이터 스케일 차이를 확인 가능
plt.figure(figsize=(10,5))

country_avg.plot(kind="bar")

plt.title("Average Fuel Price by Country", fontsize=16)
plt.xlabel("Country")
plt.ylabel("Average Fuel Price Local")

# x축 국가 이름이 겹치지 않도록 회전(미세조정)
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()
