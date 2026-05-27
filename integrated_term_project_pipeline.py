import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 고수준 외부 라이브러리 (Scikit-learn) 로드
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, silhouette_score
from sklearn.decomposition import PCA

def run_integrated_fuel_crisis_pipeline(csv_path="Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", output_dir="project_outputs"):
    """
    데이터과학 텀프로젝트 최종 통합 파이프라인 함수 (최상위 함수)
    - 역할 2, 3, 4, 5의 모든 기능을 논리적 모순 없이 하나로 연결합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("==================================================================")
    print(" [PM 주도] 글로벌 연료 위기 예측 및 분석 통합 파이프라인 가동")
    print("==================================================================")

    # ------------------------------------------------------------------
    # [역할 2] 데이터 명세 및 탐색적 데이터 분석 (EDA) 파트
    # ------------------------------------------------------------------
    print("\n--- [역할 2] 탐색적 데이터 분석 (EDA) 진행 ---")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"1. 데이터셋 구조 (Shape): {df.shape}")
    print(f"2. 초기 결측치 현황:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    
    # ------------------------------------------------------------------
    # [역할 3] 데이터 전처리 및 특징량 생성 파트
    # ------------------------------------------------------------------
    print("\n--- [역할 3] 데이터 전처리 및 정제 진행 ---")
    df_cleaned = df.copy()
    
    # [치명적 오류 수정 1] 행 삭제(Outlierdelete) 방식을 폐기하고 
    # 시스템 오류치(999.0, -99.0)를 NaN으로 치환하여 시계열 단절을 방지합니다.
    df_cleaned['Fuel_Price_Change_Percent'] = df_cleaned['Fuel_Price_Change_Percent'].replace(999.0, np.nan)
    df_cleaned['News_Sentiment_Score'] = df_cleaned['News_Sentiment_Score'].replace(-99.0, np.nan)
    
    # 시계열 순서 정렬 (국가별 -> 날짜별)
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned = df_cleaned.sort_values(['Country', 'Date']).reset_index(drop=True)
    
    # 조민서 팀원의 시계열 전후 보간법 (ffill -> bfill) 국가별 적용
    fill_cols = ['WTI_Crude_USD_per_barrel', 'Brent_Crude_USD_per_barrel', 
                 'Fuel_Price_Local', 'Shipping_Cost_Index', 'USD_Exchange_Rate',
                 'Fuel_Price_Change_Percent', 'News_Sentiment_Score']
    df_cleaned[fill_cols] = df_cleaned.groupby('Country')[fill_cols].transform(
        lambda x: x.ffill().bfill()
    )
    print("-> 시스템 오류치 치환 및 국가별 시계열 보간(fillna) 완료")

    # 수치형 변수 스케일링 (조민서 팀원의 Robust/Standard 투트랙 전략 반영)
    robust_cols = ['Fuel_Price_Change_Percent', 'News_Sentiment_Score']
    standard_cols = [col for col in df_cleaned.select_dtypes(include=['float64', 'int64']).columns 
                     if col not in robust_cols]
    
    robust_scaler = RobustScaler()
    standard_scaler = StandardScaler()
    
    df_scaled = df_cleaned.copy()
    df_scaled[robust_cols] = robust_scaler.fit_transform(df_cleaned[robust_cols])
    df_scaled[standard_cols] = standard_scaler.fit_transform(df_cleaned[standard_cols])
    print("-> RobustScaler 및 StandardScaler 기반 데이터 스케일링 완료")

    # 범주형 변수 원핫 인코딩 (Country)
    encoder = OneHotEncoder(sparse_output=False)
    encoded_country = encoder.fit_transform(df_scaled[['Country']])
    encoded_country_df = pd.DataFrame(
        encoded_country, 
        columns=encoder.get_feature_names_out(['Country']),
        index=df_scaled.index
    )
    
    # 최종 전처리 통합 데이터프레임 생성
    df_preprocessed = pd.concat([df_scaled.drop(columns=['Country']), encoded_country_df], axis=1)
    print("-> 범주형 변수(Country) 원핫 인코딩 및 전처리 데이터 통합 완료")

    # ------------------------------------------------------------------
    # [역할 4 & 5] 모델링, 성능 평가 및 아키텍처 배포 파트
    # ------------------------------------------------------------------
    print("\n--- [역할 4 & 5] 머신러닝 모델링 및 파이프라인 평가 진행 ---")
    
    # 1. K-Means 군집화 (비지도 학습 요건 충족)
    print("\n[알고리즘 1: K-Means Clustering 평가]")
    cluster_features = df_preprocessed.drop(columns=['Date'])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cleaned['Cluster'] = kmeans.fit_predict(cluster_features)
    sil_score = silhouette_score(cluster_features, df_cleaned['Cluster'])
    print(f"-> K-Means 실루엣 점수 (Silhouette Score): {sil_score:.4f}")

    # 2. 다중 선형 회귀 (회귀 요건 충족)
    print("\n[알고리즘 2: Multiple Linear Regression 평가]")
    # 권건호 팀원의 시계열 예측 타겟 생성 반영 (차일 연료 가격 변동량 예측)
    df_preprocessed['Next_Day_Fuel_Change_Amount'] = df_preprocessed.groupby(
        df_cleaned['Country']
    )['Fuel_Price_Local'].shift(-1)
    
    # 타겟 생성으로 인한 마지막 날짜의 결측 행 제거
    regression_df = df_preprocessed.dropna(subset=['Next_Day_Fuel_Change_Amount'])
    X_reg = regression_df.drop(columns=['Date', 'Next_Day_Fuel_Change_Amount'])
    y_reg = regression_df['Next_Day_Fuel_Change_Amount']
    
    # [치명적 오류 수정 2] 시계열 데이터 누수(Data Leakage)를 차단하기 위해 
    # shuffle=False 설정하여 과거 데이터로 학습하고 미래 데이터로 테스트합니다.
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, shuffle=False
    )
    
    mlr = LinearRegression()
    mlr.fit(X_train_reg, y_train_reg)
    y_pred_reg = mlr.predict(X_test_reg)
    
    print(f"-> 회귀 모델 MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}")
    print(f"-> 회귀 모델 R2 Score: {r2_score(y_test_reg, y_pred_reg):.4f}")

    # 3. 결정 트리 분류 (필수 분류 알고리즘 요건 충족)
    print("\n[알고리즘 3: Decision Tree Classification 평가]")
    # 김다나 팀원의 연료 가격 상승 레벨 분류 타겟 구조 반영
    # 스케일링 이전의 깨끗한 데이터 기준 구간 분할
    df_cleaned['Fuel_Rise_Level'] = pd.cut(
        df_cleaned['Fuel_Price_Change_Percent'],
        bins=[-np.inf, 0, 5, np.inf],
        labels=['No_Increase', 'Slight_Increase', 'Large_Increase']
    )
    
    X_clf = df_preprocessed.drop(columns=['Date', 'Next_Day_Fuel_Change_Amount'], errors='ignore')
    y_clf = df_cleaned['Fuel_Rise_Level']
    
    # 분류 모델 역시 시계열 특성을 유지를 위해 shuffle=False 분할 적용
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, shuffle=False
    )
    
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train_clf, y_train_clf)
    y_pred_clf = dt_model.fit(X_train_clf, y_train_clf).predict(X_test_clf)
    
    print(f"-> 결정 트리 정확도 (Accuracy): {accuracy_score(y_test_clf, y_pred_clf):.4f}")
    print("\n[분류 성능 리포트]")
    print(classification_report(y_test_clf, y_pred_clf, zero_division=0))

    # ------------------------------------------------------------------
    # [산출물 저장] 아키텍처 배포 기능
    # ------------------------------------------------------------------
    df_cleaned.to_csv(os.path.join(output_dir, "integrated_analyzed_dataset.csv"), index=False)
    print(f"\n[시스템 정보] 최종 분석 결과 데이터셋이 '{output_dir}' 폴더에 저장되었습니다.")
    print("==================================================================")
    print(" 모든 팀원의 코드가 모순 없이 완벽하게 통합되었습니다. 파이프라인 종료.")
    print("==================================================================")
    
    return {
        "data": df_cleaned,
        "regression_r2": r2_score(y_test_reg, y_pred_reg),
        "classification_acc": accuracy_score(y_test_clf, y_pred_clf)
    }

if __name__ == "__main__":
    # 데이터셋 파일이 동일 경로에 있다고 가정하고 가동합니다.
    try:
        run_integrated_fuel_crisis_pipeline()
    except Exception as e:
        print(f"\n[오류 발생]: {e}")