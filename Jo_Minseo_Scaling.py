import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

'''
극단적인 값(outlier)이 존재하는 컬럼에는 robust scaler를 적용하여 극단적인 값 때문에 평균값이 비정상적으로 커지는 것을 방지함. 우리의 데이터 분석의 경우, 지정학적 위기가 유가에 미치는 영향을 분석하는 것이 목적이므로 outlier를 제거하기보다는 그대로 유지하되, robust scaler로 스케일링하여 outlier의 영향을 줄이는 방법이 더 적절하다고 판단됨
'''

# 스케일링이 필요한 수치형 컬럼 확인
print("\n--- 수치형 컬럼 확인 ---")
print(df.info())

# outlier가 존재하는지 확인
# 통계적 이상치를 찾아주는 함수 정의 (IQR 방식)
def find_statistical_outliers(df, column_name):
    # 1사분위수(하위 25%)와 3사분위수(하위 75%) 계산
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    
    # IQR (상자 크기) 계산
    IQR = Q3 - Q1
    
    # 정상 범위의 최소/최대 한계선 설정 (일반적으로 1.5 * IQR을 사용)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 한계선을 벗어난 데이터(이상치)만 필터링
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    
    return outliers, lower_bound, upper_bound


# 'Date'와 'Country'를 제외한 데이터프레임 생성
df_numeric = df.drop(columns=['Date', 'Country'])
# 그 후에 숫자형 컬럼만 추출
numeric_cols = df_numeric.select_dtypes(include=['number']).columns.tolist()

# 모든 수치형 컬럼에 대해 반복문으로 이상치 탐지 실행
print(f"--- 전체 수치형 컬럼 이상치 분석 (총 {len(numeric_cols)}개 컬럼) ---")

for col in numeric_cols:
    outliers, lower, upper = find_statistical_outliers(df, col)
    
    # 이상치가 하나라도 발견되었을 때만 상세 내용을 출력
    if len(outliers) > 0:
        print(f"\n[컬럼: {col}]")
        print(f" - 발견된 이상치 개수: {len(outliers)}개")
        print(f" - 정상 범위: {lower:.2f} ~ {upper:.2f}")
    else:
        # 이상치가 없으면 짧게만 표시
        print(f"[컬럼: {col}] 이상치 없음")


print("\n--- RobustScaler 적용 전 ---")
print(df[numeric_cols].describe())

# -------------------------------------------------------------------
# 스케일링 전 데이터 복사 (원본과 그래프로 비교하기 위해)
df_original = df.copy()
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 데이터 누수 방지를 위한 Train / Test 분리
# -------------------------------------------------------------------
# 유가 데이터는 시간의 흐름이 중요한 시계열 데이터이므로 미래의 데이터로 과거를 예측하는 것을 막기 위해 섞지 않고(shuffle=False) 순서대로 자르것이 좋음
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# 스케일링 적용
robust_scaler = RobustScaler()

# Train 데이터에는 fit(기준 계산)과 transform(변환)을 동시에 적용
train_df[numeric_cols] = robust_scaler.fit_transform(train_df[numeric_cols])

# Test 데이터에는 Train에서 계산된 기준을 그대로 사용하여 transform(변환)만 적용
test_df[numeric_cols] = robust_scaler.transform(test_df[numeric_cols])

print("\n--- RobustScaler 적용 결과 (Train 기준) ---")
print(train_df[numeric_cols].describe())

# -------------------------------------------------------------------
# 스케일링 전후의 분포를 시각적으로 비교하기 위해 커널 밀도 추정(KDE) 그래프를 그리기
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. 왼쪽: 스케일링 전 (모든 변수를 하나의 축에 겹침)
for col in numeric_cols:
    sns.kdeplot(data=df_original, x=col, ax=axes[0], label=col)
axes[0].set_title('Before Scaling (Original data)')
axes[0].legend()

# 2. 오른쪽: 스케일링 후 (모든 변수를 하나의 축에 겹침)
for col in numeric_cols:
    sns.kdeplot(data=train_df, x=col, ax=axes[1], label=col)
axes[1].set_title('After Scaling (Train data)')
axes[1].legend()

plt.tight_layout()
plt.show()