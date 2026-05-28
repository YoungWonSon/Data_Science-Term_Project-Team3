import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")
date_series = pd.to_datetime(df["Date"], errors="coerce")
df["Date_Ordinal"] = date_series.map(
    lambda x: x.toordinal() if pd.notna(x) else np.nan
)
country_encoded = pd.get_dummies(df["Country"], prefix="Country", dtype=int)
numeric_cols = df.drop(columns=["Date", "Country"]).select_dtypes(
    include=["int64", "float64"]
).columns

# StandardScaler로 표준화
scaler = StandardScaler()
df_scaled_numeric = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols,
    index=df.index
)

# Date + 표준화된 숫자형 데이터 + Country one-hot 데이터 합치기
df_standardized = pd.concat(
    [
        df[["Date"]],
        df_scaled_numeric,
        country_encoded
    ],
    axis=1
)

if __name__ == "__main__":
    print("Original Data Shape:", df.shape)
    print(df.head())

    print("\nStandardized Columns:")
    print(list(numeric_cols))

    print("\nStandardized Data Shape:", df_standardized.shape)
    print(df_standardized.head())
    print(list(df_standardized.columns))


'''
#데이터 전체적 분포 확인
import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 파일 읽기
df = pd.read_csv('Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv')

# 2. 산점도 그리기
# 'X축컬럼명'과 'Y축컬럼명'을 실제 CSV 파일의 컬럼 이름으로 변경하세요.
plt.scatter(df['Date'], df['Country'], alpha=0.8)

# 3. 그래프 꾸미기 (옵션)
plt.title('Scatter Plot Title')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')

# 4. 그래프 출력
plt.show()
'''