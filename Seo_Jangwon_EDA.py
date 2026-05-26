import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 역할: Data Description & EDA (탐색적 데이터 분석)
# 목적:데이터의 구조, 결측치, 이상치, 변수 분포, 변수 간 관계를 분석하여 이후 전처리 및 머신러닝 모델링에 필요한 정보를 확인한다.


# 데이터셋 불러오기
# read_csv()를 사용하여 CSV 파일을 DataFrame 형태로 불러온다.
df = pd.read_csv(
    "/Users/seojang-won/Downloads/Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv",
    encoding="utf-8"
)


# 데이터 기본 구조 확인
# 데이터의 전체 크기(shape), 컬럼 정보(info),
# 컬럼 이름(columns)을 확인한다.
# 데이터 행(row), 열(column) 개수 확인
print("Dataset Shape:")
print(df.shape)

# 데이터 타입(dtype), 결측치 여부 확인
print("\nDataset Info:")
print(df.info())

# 전체 컬럼 이름 확인
print("\nColumns:")
print(df.columns)



# 기초 통계량 분석

# describe()를 사용하여 평균(mean), 표준편차(std),
# 최소값(min), 최대값(max) 등을 확인한다.
# 데이터의 전체적인 분포를 파악
print("\nStatistical Summary:")
print(df.describe())



# 결측치(Missing Value) 분석

# isnull().sum()을 사용하여 각 컬럼별 결측치 개수를 확인한다.
# 결측치가 있는 컬럼은 이후 전처리 과정에서
# 보간(interpolation) 또는 평균값 대체 등의 처리가 필요하다.
print("\nMissing Values:")
print(df.isnull().sum())



# 오류성 이상치(Error-type Outlier) 분석

# 일반적인 이상치(outlier)는 전쟁 및 국제 위기 상황에서는 실제 의미 있는 데이터일 가능성이 있기 때문에 제거하지 않는다.
# 대신 999.0, -99.0처럼 전산 오류 또는 비정상 입력값으로
# 보이는 값들만 별도로 확인한다.

# 999.0 개수 확인
print("\n999.0 Error Values:")
print((df == 999.0).sum())

# -99.0 개수 확인
print("\n-99.0 Error Values:")
print((df == -99.0).sum())



# 히스토그램(Histogram) 시각화

# 수치형 데이터들의 전체 분포를 확인한다.
# 히스토그램은 데이터가 어떤 구간에 많이 분포 하는지 확인함.

# figsize->그래프 전체 크기 설정
df.hist(figsize=(15, 10))

# 그래프 간 간격 자동 조정
plt.tight_layout()

# 그래프 출력
plt.show()



# 6. Fuel_Price_Change_Percent 박스플롯(Boxplot)

# 박스플롯은 데이터의 중앙값, 사분위수, 이상치를 시각적으로 확인할 때 사용한다.

# 999.0은 오류성 데이터로 판단하여 제외한다.
# 하지만 실제 큰 변동값은 위기 상황 데이터일 수 있으므로 유지한다.

# 100 미만인 데이터만 사용
filtered_data = df[df["Fuel_Price_Change_Percent"] < 100]


# figsize->그래프 크기 설정
plt.figure(figsize=(8, 4))

# 박스플롯 생성
sns.boxplot(
    x=filtered_data["Fuel_Price_Change_Percent"],

    # 이상치(outlier) 점 스타일 설정(빨간색)
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "red",
        "markersize": 4
    }
)

# 제목 설정
plt.title("Fuel Price Change Percent Boxplot", fontsize=15)

# x축 이름 설정
plt.xlabel("Fuel Price Change Percent")

# 레이아웃 자동 조정
plt.tight_layout()

# 그래프 출력
plt.show()



# 상관관계 히트맵(Heatmap)

# corr()를 사용하여 변수 간 상관관계를 계산한다.
# 상관계수:1에 가까울수록 강한 양의 상관관계,-1에 가까울수록 강한 음의 상관관계,0에 가까울수록 관계가 거의 없음
# heatmap은 변수들 사이의 관계를 색상으로 표현한다.

# 히트맵 크기 설정
plt.figure(figsize=(13, 9))

# 히트맵 생성
sns.heatmap(

    # 수치형 컬럼만 사용하여 상관관계 계산
    df.corr(numeric_only=True),

    # 각 칸에 숫자 표시
    annot=True,

    # 색상 테마 설정
    cmap="coolwarm",

    # 소수점 둘째 자리까지 표시
    fmt=".2f"
)

# x축 글씨 회전 및 크기 설정
plt.xticks(rotation=45, ha="right", fontsize=8)

# y축 글씨 크기 설정
plt.yticks(rotation=0, fontsize=9)

# 제목 설정
plt.title("Correlation Heatmap", fontsize=16)

# 레이아웃 자동 조정
plt.tight_layout()

# 그래프 출력
plt.show()



# 국가별 연료 가격 박스플롯
# 국가별 Fuel_Price_Local 값의 분포 차이를 확인한다.
# 국가마다 값 차이가 매우 크기 때문에 y축에 로그 스케일(log scale)을 적용하여 작은 값들도 잘 보이도록 설정한다.

# 그래프 크기 설정
plt.figure(figsize=(14, 7))

# 국가별 박스플롯 생성
sns.boxplot(
    x="Country",
    y="Fuel_Price_Local",
    data=df,

    # 이상치 점 스타일 설정
    flierprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "red",
        "markersize": 4
    }
)

# 로그 스케일 적용
plt.yscale("log")

# 제목 설정
plt.title("Fuel Price Distribution by Country", fontsize=16)

# x축 이름 설정
plt.xlabel("Country")

# y축 이름 설정
plt.ylabel("Fuel Price Local (Log Scale)")

# 국가 이름 회전
plt.xticks(rotation=45, ha="right")

# 레이아웃 자동 조정
plt.tight_layout()

# 그래프 출력
plt.show()


# 상관관계 높은 변수 조합 확인
#히트맵은 변수 개수가 많아지면 보기 어려울 수 있으므로,상관관계가 높은 변수 조합만 따로 정렬하여 출력한다.

# 절대값 기준 상관관계 계산
corr_matrix = df.corr(numeric_only=True).abs()

# 상관관계를 한 줄 형태로 변환 후 정렬
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# 자기 자신과의 상관관계(1.0)는 제외
corr_pairs = corr_pairs[corr_pairs < 1]

# 상관관계 높은 상위 10개 출력
print("\nTop 10 Highly Correlated Feature Pairs:")
print(corr_pairs.head(10))
