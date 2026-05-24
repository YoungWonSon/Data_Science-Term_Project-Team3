import pandas as pd

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

# 각 컬럼별 결측치 개수와 결측치가 존재하는 컬럼 확인
print("\nMissing Values:")
print(df.isnull().sum())

# 결측치가 존재하는 컬럼 리스트
cols = ['WTI_Crude_USD_per_barrel', 'Brent_Crude_USD_per_barrel', 
        'Fuel_Price_Local', 'Shipping_Cost_Index', 'USD_Exchange_Rate']

# Date 컬럼을 datetime 형식으로 변환, object 타입이어도 ffill 적용되긴 하지만 시계열 데이터임을 확실히 하기 위해 변환
df['Date'] = pd.to_datetime(df['Date'])

# 국가별, 날짜별 정렬
df = df.sort_values(['Country', 'Date'])

# 사용하는 데이터 셋은 시간의 흐름에 따른 시계열 데이터이므로, 결측치를 채울 때는 앞뒤의 값으로 채우는 방법을 사용 
#groupby('Country')로 국가별로 그룹화한 후, transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))로 각 그룹에 대해 먼저 ffill을 적용하여 이전 값으로 결측치를 채우고, 앞의 값으로 채워지지 않는 결측치는 bfill로 뒤의 값으로 채움
df[cols] = df.groupby('Country')[cols].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

# 각 컬럼별 결측치 개수와 결측치가 존재하는 컬럼 확인
print("\nMissing Values:")
print(df.isnull().sum())

# 변환된 데이터프레임 확인
print(df)