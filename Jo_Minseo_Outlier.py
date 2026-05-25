import pandas as pd
import numpy as np

df = pd.read_csv("Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", encoding="utf-8")

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

# 수치형 데이터만 자동으로 추출 (float, int 타입만 선택)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

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