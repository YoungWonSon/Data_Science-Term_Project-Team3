import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 불러오기
df = pd.read_csv("/Users/seojang-won/Downloads/Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

# 1. 데이터 기본 정보 확인

# 데이터 크기 출력
print("Dataset Shape:")
print(df.shape)

# 데이터 정보 출력
print("\nDataset Info:")
print(df.info())

# 컬럼 이름 출력
print("\nColumns:")
print(df.columns)

# 2. 기초 통계량 분석

# 평균, 표준편차, 최소값, 최대값 출력
print("\nStatistical Summary:")
print(df.describe())

# 3. 결측치 분석

# 각 컬럼별 결측치 개수 확인
print("\nMissing Values:")
print(df.isnull().sum())

# 4. 이상치 분석

# 999.0 이상치 개수 확인
print("\n999.0 Outliers:")
print((df == 999.0).sum())

# -99.0 이상치 개수 확인
print("\n-99.0 Outliers:")
print((df == -99.0).sum())

# 5. 히스토그램 시각화

# 전체 수치형 데이터 분포 확인
df.hist(figsize=(15,10))
plt.tight_layout()
plt.show()

# 6. 박스플롯 시각화

# Fuel_Price_Change_Percent 이상치 확인
plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fuel_Price_Change_Percent"])

plt.title("Fuel Price Change Percent Boxplot")
plt.show()

# 7. 상관관계 히트맵

# 수치형 변수 간 상관관계 분석
plt.figure(figsize=(13,9))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

# x축 글씨 회전 및 크기 조절
plt.xticks(rotation=45, ha='right', fontsize=8)

# y축 글씨 크기 조절
plt.yticks(rotation=0, fontsize=9)

# 제목
plt.title("Correlation Heatmap", fontsize=16)

# 자동 여백 조정
plt.tight_layout()

plt.show()
