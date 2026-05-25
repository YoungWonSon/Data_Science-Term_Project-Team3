import pandas as pd

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

print("--- [before] 데이터 행 개수 ---")
print(len(df))

# 이상치 확인
print("\n--- 이상치가 존재하는 컬럼 확인 ---")
print("[999.0 이상치]")
print((df == 999.0).sum())
print("\n[-99.0 이상치]")
print((df == -99.0).sum())

# 이전 분석에서 확인했던 특정 컬럼의 이상치를 필터링하여 정상 데이터만 남깁니다.
df_cleaned = df[(df['Fuel_Price_Change_Percent'] != 999.0) & (df['News_Sentiment_Score'] != -99.0)]

print("\n--- [after] 데이터 행 개수 ---")
print(len(df_cleaned))

# 이상치 확인
print("\n--- 이상치가 존재하는 컬럼 확인 ---")
print("[999.0 이상치]")
print((df_cleaned == 999.0).sum())
print("\n[-99.0 이상치]")
print((df_cleaned == -99.0).sum())
