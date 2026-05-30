import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import Kim_Dana_Standarization_Data

df_standardized = Kim_Dana_Standarization_Data.df_standardized.copy()
raw_df = Kim_Dana_Standarization_Data.df.copy()

# 1. Clustering용 데이터 준비
# Date는 문자열 데이터이므로 K-Means에 사용할 수 없음
#X = df_standardized.drop(columns=["Date"], errors="ignore")
#X = df_standardized.drop(columns=["Date", "Date_Ordinal"], errors="ignore")

clustering_cols_simple = [
    'Brent_Crude_USD_per_barrel',
    'Fuel_Price_Local',
    'Fuel_Price_Change_Percent',
    'Crisis_Intensity_Index',
    'USD_Exchange_Rate',
    'Inflation_Rate_Percent'
]
X = df_standardized[clustering_cols_simple]

# 혹시 object 타입 컬럼이 남아 있을 경우 제거
X = X.select_dtypes(include=["int64", "float64", "int32", "float32", "uint8", "bool"])

# K-Means는 결측치를 처리할 수 없으므로 결측치가 있는 행 제거 -> 제거된 행과 원본 df의 index를 맞춤
X_clean = X.dropna()
df_result = df_standardized.loc[X_clean.index].copy()

print(df_result.head())
print("Clustering Data Shape:", X_clean.shape)

# 2. 최적 k 찾기
k_results = []
max_k = len(X_clean.columns) * 2
print("max_k: ", max_k)

for k in range(2, max_k):
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
print("\nK-Means Evaluation Results Sorted by Silhouette Score:")
k_results_sorted = k_results_df.sort_values(
    by="silhouette_score",
    ascending=False
)
print(k_results_sorted)

# Silhouette Score가 가장 높은 k 선택
best_k = 8
#best_k = int(k_results_df.loc[k_results_df["silhouette_score"].idxmax(), "n_clusters"])
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

# 4. Elbow Method 그래프
plt.figure(figsize=(8, 5))
plt.plot(k_results_df["n_clusters"], k_results_df["inertia"], marker="o")
plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

# 5. Silhouette Score 그래프
plt.figure(figsize=(8, 5))
plt.plot(k_results_df["n_clusters"], k_results_df["silhouette_score"], marker="o")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# 7. Cluster Center 확인
cluster_centers = pd.DataFrame(
    best_kmeans.cluster_centers_,
    columns=X_clean.columns
)
print("\nCluster Centers:")
print(cluster_centers.round(4))

# 원본 데이터 기준으로 클러스터 결과 붙이기
raw_clustered_df = raw_df.loc[X_clean.index].copy()
raw_clustered_df["Cluster"] = df_result["Cluster"].values
raw_clustered_df["Date"] = raw_clustered_df["Date_Ordinal"].apply(
    lambda x: pd.Timestamp.fromordinal(int(x)))

print(raw_clustered_df[["Date", "Country", "Cluster"]].head())

# Country를 y축에 표시하기 위해 숫자로 변환
country_order = sorted(raw_clustered_df["Country"].dropna().unique())
country_to_num = {country: i for i, country in enumerate(country_order)}
raw_clustered_df["Country_Num"] = raw_clustered_df["Country"].map(country_to_num)

plt.figure(figsize=(14, 7))

scatter = plt.scatter(
    raw_clustered_df["Date"],
    raw_clustered_df["Country_Num"],
    c=raw_clustered_df["Cluster"],
    alpha=0.8
)

plt.yticks(
    ticks=list(country_to_num.values()),
    labels=list(country_to_num.keys())
)

plt.title("K-Means Clustering Result by Date and Country")
plt.xlabel("Date")
plt.ylabel("Country")

cbar = plt.colorbar(scatter)
cbar.set_label("Cluster")

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

########################################################## 이 밑으로는 추가적인 결과를 원하시면, 주석을 풀고 체크하시면 됩니다.

# 6. PCA를 이용한 2D 시각화
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)
centers_pca = pca.transform(best_kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df_result["Cluster"],
    alpha=0.7
)

plt.scatter(
    centers_pca[:, 0],
    centers_pca[:, 1],
    c=np.arange(best_k),
    cmap="viridis",
    vmin=0,
    vmax=best_k - 1,
    marker=".",
    s=300,
    edgecolors="black",
    linewidths=1.5,
    label="Cluster Centers"
)

plt.title(f"K-Means Clustering Result with PCA 2D Visualization, k={best_k}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
######################################################
'''
1. Cluster 0: Brent 높음, Fuel Price 높음, Crisis 높음, USD 높음, Inflation 약간 낮음
      고유가 + 높은 위기 강도 + 환율 상승형 클러스터 = 이란 전쟁/위기 상황으로 인해 유가가 상승한 전형적인 위기성 고유가 시장 해석

2. Cluster 1: Brent 낮음, Fuel Price 낮음, Fuel Change 낮음, Crisis 낮음, USD 낮음, Inflation 낮음
                안정적인 저유가 시장 클러스터
3. Cluster 2: Brent 낮음, Fuel Price 매우 높음, USD 매우 높음, Inflation 높음, Crisis 낮음
                국제 유가는 낮지만 현지 연료 가격이 높은 환율·인플레이션 영향형 클러스터
4. Cluster 3: Fuel_Price_Change_Percent = 19.3138, 다른 변수들은 대부분 평균 이하 또는 평균 근처
                연료 가격 변화율 이상치 클러스터
5. Cluster 4: Brent 높음, Crisis 높음, Fuel Price 낮음, USD 낮음, Inflation 낮음
                국제 유가는 상승했지만, 환율이나 인플레이션 압력이 낮아서 현지 연료 가격까지 크게 상승하지 않은 그룹
6. Cluster 5: Brent 낮음, Crisis 낮음, Fuel Price 약간 높음, USD 높음, Inflation 낮음
                국제 유가와 위기 강도는 낮지만 환율 때문에 현지 연료 가격이 오른 클러스터
7. Cluster 6: Brent 높음, Crisis 높음, Inflation 매우 높음, Fuel Price 거의 평균
                고유가 + 높은 위기 강도 + 매우 높은 인플레이션 클러스터, 유가 위기와 인플레이션 압력이 동시에 강한 시장 상황
8. Cluster 7: Brent 매우 높음, Fuel Price 매우 높음, Crisis 높음, USD 매우 높음, Inflation 높음
                가장 심각한 고유가·위기·환율·인플레이션 복합 충격 클러스터
9. Cluster 8: Brent 낮음, Fuel Price 낮음, Crisis 낮음, Inflation 매우 높음
                유가 위기는 약하지만 인플레이션이 높은 클러스터, 일반적인 물가 상승 압력이 강한 그룹으로 해석

'''
'''
cluster_centers = pd.DataFrame(
    best_kmeans.cluster_centers_,
    columns=X_clean.columns
)
print("\nCluster Centers:")
print(cluster_centers.round(4))

# 8. 원본 데이터 기준 Cluster별 평균 확인
numeric_original_cols = df_standardized.drop(columns=["Date", "Country"], errors="ignore").select_dtypes(
    include=["int64", "float64"]
).columns
cluster_summary = df_result.groupby("Cluster")[numeric_original_cols].mean()

print("\nCluster Summary Based on Original Numeric Values:")
print(cluster_summary.round(4))
'''