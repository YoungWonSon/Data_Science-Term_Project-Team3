import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_manual_integrated_pipeline(csv_path="Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv", output_dir="manual_outputs"):
    """
    [PM 주도] 고수준 라이브러리(Scikit-learn)를 완전히 배제하고,
    오직 NumPy와 Pandas의 수학/배열 연산만으로 구현한 독자적인 엔드투엔드 파이프라인.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("==================================================================")
    print(" [PM 주도 - Manual Version] 순수 NumPy/Pandas 기반 파이프라인 가동")
    print("==================================================================")

    # ------------------------------------------------------------------
    # [역할 2] 데이터 로드 및 확인
    # ------------------------------------------------------------------
    print("\n--- [역할 2] 데이터 로드 ---")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"데이터셋 구조: {df.shape}")
    
    # ------------------------------------------------------------------
    # [역할 3] 데이터 전처리 (수동 구현 버전)
    # ------------------------------------------------------------------
    print("\n--- [역할 3] 데이터 전처리 (고수준 모듈 배제) ---")
    df_cleaned = df.copy()
    
    # 이상치 처리: 행 삭제 방지 및 NaN 치환
    df_cleaned['Fuel_Price_Change_Percent'] = df_cleaned['Fuel_Price_Change_Percent'].replace(999.0, np.nan)
    df_cleaned['News_Sentiment_Score'] = df_cleaned['News_Sentiment_Score'].replace(-99.0, np.nan)
    
    # 시계열 정렬
    df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])
    df_cleaned = df_cleaned.sort_values(['Country', 'Date']).reset_index(drop=True)
    
    # 시계열 보간 (Pandas 내장 메서드 활용)
    fill_cols = ['WTI_Crude_USD_per_barrel', 'Brent_Crude_USD_per_barrel', 
                 'Fuel_Price_Local', 'Shipping_Cost_Index', 'USD_Exchange_Rate',
                 'Fuel_Price_Change_Percent', 'News_Sentiment_Score']
    df_cleaned[fill_cols] = df_cleaned.groupby('Country')[fill_cols].transform(
        lambda x: x.ffill().bfill()
    )
    print("-> 결측치 처리 및 보간 완료")

    # 1. 수동 스케일링 구현 (StandardScaler 대체)
    # 변경점: sklearn.preprocessing.StandardScaler() 대신 통계 공식 (x - mean) / std 를 직접 연산합니다.
    df_scaled = df_cleaned.copy()
    numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        mean_val = df_scaled[col].mean()
        std_val = df_scaled[col].std()
        if std_val != 0:  # 0으로 나누기 방지
            df_scaled[col] = (df_scaled[col] - mean_val) / std_val
    print("-> 수학 공식을 활용한 수동 표준화(Standardization) 완료")

    # 2. 수동 원핫 인코딩 구현 (OneHotEncoder 대체)
    # 변경점: sklearn.preprocessing.OneHotEncoder() 대신 Pandas의 배열 생성 기능(get_dummies)을 활용합니다.
    df_encoded = pd.get_dummies(df_scaled, columns=['Country'], dtype=float)
    print("-> Pandas 배열 기반 수동 범주형 변수 인코딩 완료")

    # ------------------------------------------------------------------
    # [역할 4 & 5] 수동 알고리즘 구현 및 평가
    # ------------------------------------------------------------------
    print("\n--- [역할 4 & 5] 수동 알고리즘 기반 모델링 진행 ---")
    
    # ------------------------------------------------------------------
    # 모델 1: 수동 K-Nearest Neighbors (KNN) 분류 (결정 트리 대체)
    # 변경점: 복잡한 Tree 라이브러리 대신, 유클리디안 거리를 NumPy 행렬 연산으로 계산하는 KNN을 직접 구현
    print("\n[알고리즘 1: 수동 KNN 분류]")
    
    # 타겟 생성 (유가 상승 레벨 - 원본 데이터 기준)
    y_class = pd.cut(
        df_cleaned['Fuel_Price_Change_Percent'],
        bins=[-np.inf, 0, 5, np.inf],
        labels=[0, 1, 2] # 연산을 위해 숫자로 인코딩 (0:하락/유지, 1:소폭상승, 2:대폭상승)
    ).astype(float) # NaN 처리를 위해 float으로 변환
    
    # 결측 타겟 제거 및 데이터 분할
    valid_idx = ~y_class.isna()
    X_clf = df_encoded.drop(columns=['Date'])[valid_idx].values
    y_clf = y_class[valid_idx].values
    
    # 시계열 순차 분할 (미래 데이터 누수 차단)
    split_idx_clf = int(len(X_clf) * 0.8)
    X_train_clf, X_test_clf = X_clf[:split_idx_clf], X_clf[split_idx_clf:]
    y_train_clf, y_test_clf = y_clf[:split_idx_clf], y_clf[split_idx_clf:]

    def manual_knn_predict(X_train, y_train, X_test, k=5):
        """NumPy 배열 브로드캐스팅을 활용한 유클리디안 거리 계산 기반 KNN"""
        y_pred = []
        for test_point in X_test:
            # 1. 모든 훈련 데이터와 테스트 포인트 간의 거리 계산 (유클리디안 거리)
            distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
            # 2. 거리가 가장 가까운 k개의 인덱스 추출
            nearest_indices = np.argsort(distances)[:k]
            # 3. 이웃들의 레이블 중 최빈값(다수결) 선택
            nearest_labels = y_train[nearest_indices]
            unique, counts = np.unique(nearest_labels, return_counts=True)
            y_pred.append(unique[np.argmax(counts)])
        return np.array(y_pred)

    y_pred_clf = manual_knn_predict(X_train_clf, y_train_clf, X_test_clf, k=5)
    acc = np.sum(y_pred_clf == y_test_clf) / len(y_test_clf)
    print(f"-> 수동 KNN 분류 정확도 (Accuracy): {acc:.4f}")

    # ------------------------------------------------------------------
    # 모델 2: 수동 다중 선형 회귀 (LinearRegression 대체)
    # 변경점: 선형대수학의 정규 방정식(Normal Equation) 𝛉 = (X^T * X)^-1 * X^T * y 을 직접 계산
    print("\n[알고리즘 2: 정규 방정식(Normal Equation) 기반 수동 다중 선형 회귀]")
    
    # 타겟 생성 (다음 날 연료 가격 변동량)
    df_encoded['Next_Day_Change'] = df_encoded.groupby(
        df_cleaned['Country']
    )['Fuel_Price_Local'].shift(-1)
    
    regression_df = df_encoded.dropna(subset=['Next_Day_Change'])
    X_reg = regression_df.drop(columns=['Date', 'Next_Day_Change']).values
    y_reg = regression_df['Next_Day_Change'].values
    
    # 시계열 순차 분할
    split_idx_reg = int(len(X_reg) * 0.8)
    X_train_reg, X_test_reg = X_reg[:split_idx_reg], X_reg[split_idx_reg:]
    y_train_reg, y_test_reg = y_reg[:split_idx_reg], y_reg[split_idx_reg:]

    # 바이어스(편향) 항을 위해 1로 채워진 열 추가
    X_train_reg_bias = np.c_[np.ones((X_train_reg.shape[0], 1)), X_train_reg]
    X_test_reg_bias = np.c_[np.ones((X_test_reg.shape[0], 1)), X_test_reg]

    # 정규 방정식 연산: theta_best = inv(X.T @ X) @ X.T @ y
    # np.linalg.inv (역행렬 계산) 과 행렬 곱셈(@) 만을 활용
    theta_best = np.linalg.inv(X_train_reg_bias.T.dot(X_train_reg_bias)).dot(X_train_reg_bias.T).dot(y_train_reg)
    
    # 예측 수행
    y_pred_reg = X_test_reg_bias.dot(theta_best)
    
    # 수동 MSE 계산: 오차 제곱의 평균
    mse = np.mean((y_test_reg - y_pred_reg) ** 2)
    print(f"-> 수동 회귀 모델 MSE: {mse:.4f}")

    # ------------------------------------------------------------------
    # [산출물 저장] 
    # ------------------------------------------------------------------
    df_cleaned.to_csv(os.path.join(output_dir, "manual_analyzed_dataset.csv"), index=False)
    print(f"\n[시스템 정보] 최종 분석 결과 데이터셋이 '{output_dir}' 폴더에 저장되었습니다.")
    print("==================================================================")
    print(" 고수준 모듈 없이 순수 수학 연산으로 파이프라인 통합 완료.")
    print("==================================================================")
    
    return {
        "manual_knn_accuracy": acc,
        "manual_regression_mse": mse
    }

if __name__ == "__main__":
    run_manual_integrated_pipeline()