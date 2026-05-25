import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

'''
극단적인 값(이상치)가 존재하는 컬럼에는 robust scaler를 적용하여 극단적인 값 떄문에 평균값이 비정상적으로 커지는 것을 방지하고, 극단적인 값이 존재하지 않는 수치형 컬럼에는 standard scaler를 적용하여 평균이 0, 표준편차가 1이 되도록 스케일링해서 각 값들이 평균에 비해 얼마나 높고 낮은지를 나타내도록 한다.
'''

# 스케일링이 필요한 수치형 컬럼 확인
print("\n--- 수치형 컬럼 확인 ---")
print(df.info())
#robust scaler 적용하기 위해 이상치가 존재하는 컬럼 확인
print("\n--- 이상치가 존재하는 컬럼 확인 ---")
print((df == 999.0).sum())
print((df == -99.0).sum())

# RobustScaler 적용을 위해 이상치가 존재하는 컬럼 지정
robust_cols = ['Fuel_Price_Change_Percent', 'News_Sentiment_Score']

# StandardScaler 적용을 위해 이상치가 존재하지 않는 수치형 컬럼 지정
# 숫자형 컬럼을 먼저 선택한 후, 이상치가 존재하는 컬럼을 제외하여 standard scaler 적용할 컬럼 리스트 생성
numeric_cols = df.select_dtypes(include=['float64']).columns.tolist()
standard_cols = [col for col in numeric_cols if col not in robust_cols]

print("\n--- RobustScaler 적용 전 ---")
print(df[robust_cols].describe())

print("\n--- StandardScaler 적용 전 ---")
print(df[standard_cols].describe())

# -------------------------------------------------------------------
# 스케일링 전 데이터 복사 (원본과 그래프로 비교하기 위해)
df_original = df.copy()
# -------------------------------------------------------------------

# 스케일링 적용
robust_scaler = RobustScaler()
standard_scaler = StandardScaler()

df[robust_cols] = robust_scaler.fit_transform(df[robust_cols])
df[standard_cols] = standard_scaler.fit_transform(df[standard_cols])

print("\n--- RobustScaler 적용 결과 ---")
print(df[robust_cols].describe())

print("\n--- StandardScaler 적용 결과 ---")
print(df[standard_cols].describe())

# -------------------------------------------------------------------
# 스케일링 전후의 분포를 시각적으로 비교하기 위해 커널 밀도 추정(KDE) 그래프를 그리기
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1. 왼쪽: 스케일링 전 (모든 변수를 하나의 축에 겹침)
for col in (robust_cols + standard_cols):
    sns.kdeplot(data=df_original, x=col, ax=axes[0], label=col)
axes[0].set_title('Before Scaling')
axes[0].legend()

# 2. 오른쪽: 스케일링 후 (모든 변수를 하나의 축에 겹침)
for col in (robust_cols + standard_cols):
    sns.kdeplot(data=df, x=col, ax=axes[1], label=col)
axes[1].set_title('After Scaling')
axes[1].legend()

plt.tight_layout()
plt.show()