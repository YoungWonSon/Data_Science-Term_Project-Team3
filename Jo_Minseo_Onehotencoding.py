import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

# 각 feature의 데이터타입 확인
print(df.dtypes)

#Country feature는 순위가 매겨지지 않으므로 label encoding이 아닌 one-hot encoding을 사용해야 함

#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#OneHotEncoder(): 범주형 데이터를 0과 1로 이루어진 벡터로 변환, Country는 순위가 정해져 있지 않으므로 label encoding이 아닌 one-hot encoding을 사용해야 함

#sparse_output=False: 결과를 희소 행렬이 아닌 일반 배열로 반환하도록 설정
#희소행렬: 대부분의 값이 0인 행렬을 효율적으로 저장하는 방법, 메모리 절약과 계산 속도 향상에 도움
encoder = OneHotEncoder(sparse_output=False)

#OneHotEncoder는 2차원 배열을 입력으로 받기 때문에, df[['Country']]와 같이 2차원으로 입력
encoded_array = encoder.fit_transform(df[['Country']])

#encoded_array는 numpy 배열이므로, 이를 DataFrame으로 변환하여 원래의 데이터프레임과 병합할 수 있음
encoded_df=pd.DataFrame(encoded_array)

 # 원래 feature 이름을 기반으로  column 이름 변경
encoded_df.columns = encoder.get_feature_names_out(['Country'])

#원본 데이터에 병합
df = pd.concat([df, encoded_df], axis=1)

#원본 feature 제거
df=df.drop(columns=['Country'])

#변환된 데이터프레임 확인
print(df.columns)
print(df)

