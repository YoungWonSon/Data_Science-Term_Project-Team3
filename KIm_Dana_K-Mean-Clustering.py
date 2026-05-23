import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from Kim_Dana_Standarization_Data import df, df_standardized
'''
k-Means Clustering vs Hierarchical Agglomerative Clustering

1. 숫자형 변수는 표준화, Country는 one-hot encoding = 이런 구조는 K-Means가 centroid를 기준으로 패턴을 나누기에 적합한 형태
2. K-Means: 클러스터 중심점인 centroid를 기준으로 데이터를 반복적으로 분류.
            => 데이터 개수가 많아도 비교적 빠르게 실행
   Hierarchical Agglomerative Clustering: 처음에 각 데이터를 하나의 클러스터로 보고, 가장 가까운 클러스터끼리 계속 병합
                                       => 전체 거리 관계를 많이 계산해야 하기 때문에 데이터가 많아질수록 느림
3. K-Means: 각 클러스터의 중심값 확인 가능 => K-Means는 클러스터별 특징을 설명하기 쉬움
   Hierarchical Clustering: dendrogram을 통해 데이터가 어떻게 병합되는지는 볼 수 있지만, 
                            최종 클러스터의 특징을 직관적으로 해석하려면 추가 분석이 필요
                            
inertia = K-Means가 만든 클러스터가 얼마나 잘 모여 있는지를 나타내는 값
          각 데이터가 자기 클러스터의 중심점에서 얼마나 떨어져 있는지의 총합
inertia가 작다 = 데이터들이 중심점 주변에 잘 모여 있다
inertia가 크다 = 데이터들이 중심점에서 많이 흩어져 있다

Elbow Method: 여러 개의 k 값을 실험해 보고, 각 k의 inertia를 그래프로 그린 뒤 적절한 클러스터 개수를 고르는 방법
              클러스터 수를 늘렸을 때 성능 향상이 갑자기 줄어드는 지점을 찾는 방법

Silhouette Score: 클러스터링이 얼마나 잘 되었는지를 평가하는 점수
                  1) Cluster 안에서 잘 뭉쳐 있는가?
                  2) 다른 Cluster와 잘 구분되는가?
                  => 내 클러스터 안에서는 가깝고, 다른 클러스터와는 멀어야 좋다.
                1에 가까움 = 클러스터가 매우 잘 나뉨
'''

# 1. Clustering용 데이터 준비
# Date는 문자열 데이터이므로 K-Means에 사용할 수 없음
X = df_standardized.drop(columns=["Date"], errors="ignore")

# 혹시 object 타입 컬럼이 남아 있을 경우 제거
X = X.select_dtypes(include=["int64", "float64", "int32", "float32", "uint8", "bool"])

# K-Means는 결측치를 처리할 수 없으므로 결측치가 있는 행 제거 -> 제거된 행과 원본 df의 index를 맞춤
X_clean = X.dropna()
df_result = df.loc[X_clean.index].copy()

print("Clustering Data Shape:", X_clean.shape)
print("Columns used for clustering:")
print(list(X_clean.columns))

# 2. 최적 k 찾기
k_results = []
max_k = min(10, len(X_clean) - 1)

for k in range(2, max_k + 1):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_clean)
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_clean, labels)

    k_results.append({
        "n_clusters": k,
        "inertia": inertia,
        "silhouette_score": silhouette
    })

k_results_df = pd.DataFrame(k_results)
print("\nK-Means Evaluation Results:")
print(k_results_df)

# Silhouette Score가 가장 높은 k 선택
best_k = int(k_results_df.loc[k_results_df["silhouette_score"].idxmax(), "n_clusters"])
print("\nBest number of clusters:", best_k)

# 3. Best K-Means Model 학습
best_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)
df_result["Cluster"] = best_kmeans.fit_predict(X_clean)

print("\nCluster Counts:")
print(df_result["Cluster"].value_counts().sort_index())

# 4. Cluster Center 확인
# 상대적으로 원유 가격이 낮은 시기 또는 안정적인 시장 상황을 나타내는 그룹 vs
#                                 원유 가격이 크게 상승한 고유가 상황 또는 위기성이 강한 시장 상황을 나타내는 그룹
cluster_centers = pd.DataFrame(
    best_kmeans.cluster_centers_,
    columns=X_clean.columns
)
print("\nCluster Centers:")
print(cluster_centers.round(4))

# 5. 원본 데이터 기준 Cluster별 평균 확인
numeric_original_cols = df.drop(columns=["Date", "Country"], errors="ignore").select_dtypes(
    include=["int64", "float64"]
).columns
cluster_summary = df_result.groupby("Cluster")[numeric_original_cols].mean()

print("\nCluster Summary Based on Original Numeric Values:")
print(cluster_summary.round(4))

# 6. Elbow Method 그래프
plt.figure(figsize=(8, 5))
plt.plot(k_results_df["n_clusters"], k_results_df["inertia"], marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 7. Silhouette Score 그래프
plt.figure(figsize=(8, 5))
plt.plot(k_results_df["n_clusters"], k_results_df["silhouette_score"], marker="o")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# 8. PCA를 이용한 2D 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df_result["Cluster"],
    alpha=0.7
)

plt.title(f"K-Means Clustering Result with PCA 2D Visualization, k={best_k}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()