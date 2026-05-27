import pandas as pd
import numpy as np

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

# 오류성 이상치 확인
print("\n--- 오류성 이상치가 존재하는 컬럼 확인 ---")
print("[999.0 이상치]")
print((df == 999.0).sum())
print("\n[-99.0 이상치]")
print((df == -99.0).sum())

# 이전 분석에서 확인했던 특정 컬럼의 오류성 이상치를 필터링하여 정상 데이터만 남깁니다.
df['Fuel_Price_Change_Percent'] = df['Fuel_Price_Change_Percent'].replace(999.0, np.nan)
df['News_Sentiment_Score'] = df['News_Sentiment_Score'].replace(-99.0, np.nan)

# 오류성 이상치 확인
print("\n--- 오류성 이상치가 존재하는 컬럼 확인 ---")
print("[999.0 이상치]")
print((df == 999.0).sum())
print("\n[-99.0 이상치]")
print((df == -99.0).sum())
