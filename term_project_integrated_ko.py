import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# Team 3 Term Project Integrated Pipeline
# Korean-comment version
# ============================================================
# 이 파일은 지금까지 받은 팀원 코드들을 하나의 실행 흐름으로 통합한 버전이다.
#
# 반영한 팀원별 작업:
# 1. 2번 역할(EDA): 데이터 구조, 결측치, 이상치, 통계량, 상관관계, 국가별 분포 확인
# 2. 3번 역할(전처리): 결측치 처리, 오류성 이상치 처리, Country 인코딩, Robust scaling
# 3. 4번 역할(모델링): Decision Tree classification, Ridge regression, K-Means clustering
#
# PM 통합 기준:
# - A안 유지: Fuel_Price_Change_Percent를 기준으로
#   No_Increase / Slight_Increase / Large_Increase 3-class classification 수행
# - 4번 역할 요청사항 반영: train_test_split에서 shuffle=True 사용
# - 3번 역할 요청사항 반영: scaling은 RobustScaler만 사용
# - 오류성 dirty value 처리:
#   Fuel_Price_Change_Percent > 100, 특히 999.0은 전산 오류로 보고 NaN 처리
#   News_Sentiment_Score == -99.0은 전산 오류로 보고 NaN 처리
# - 데이터 누수 방지:
#   scaler와 imputer는 전체 데이터에 미리 fit하지 않고,
#   train set에만 fit한 뒤 test set에는 transform만 적용한다.
#
# ------------------------------------------------------------
# 원 팀원 코드의 주석/의도를 통합한 상세 설명
# ------------------------------------------------------------
# 2번 역할(데이터 명세 및 EDA)에서 강조한 내용:
# - EDA의 목적은 모델링 전에 데이터의 구조, 결측치, 이상치, 변수 분포,
#   변수 간 관계를 분석하여 이후 전처리와 머신러닝 모델링에 필요한 정보를
#   확인하는 것이다.
# - df.shape는 데이터 행(row)과 열(column)의 개수를 확인하기 위해 사용한다.
# - df.info()는 각 feature의 dtype과 non-null 개수를 확인하기 위해 사용한다.
# - df.describe()는 평균(mean), 표준편차(std), 최솟값(min), 최댓값(max),
#   사분위수 등을 확인해 데이터의 전체적인 분포를 파악하기 위해 사용한다.
# - isnull().sum()은 컬럼별 결측치 개수를 확인하기 위해 사용한다.
# - 999.0, -99.0처럼 정상 범위를 크게 벗어난 값은 전쟁/위기 상황의
#   실제 극단값이 아니라 전산 오류 또는 비정상 입력값으로 보고 별도로 처리한다.
# - histogram은 각 수치형 변수가 어떤 구간에 많이 분포하는지 확인하기 위해 사용한다.
# - boxplot은 중앙값, 사분위수, 이상치를 시각적으로 확인하기 위해 사용한다.
# - correlation heatmap은 변수 간 상관관계를 색상으로 표현해,
#   1에 가까울수록 강한 양의 상관관계, -1에 가까울수록 강한 음의 상관관계,
#   0에 가까울수록 관계가 약하다는 점을 보여준다.
# - 국가별 Fuel_Price_Local은 국가마다 화폐 단위와 가격 규모가 매우 다르므로
#   log scale boxplot을 사용하면 작은 값과 큰 값을 동시에 보기 쉽다.
#
# 3번 역할(전처리 및 특징량 생성)에서 강조한 내용:
# - 결측치가 존재하는 핵심 컬럼은 WTI, Brent, Fuel_Price_Local,
#   Shipping_Cost_Index, USD_Exchange_Rate이다.
# - 데이터가 날짜 흐름을 가진 국가별 시계열 형태이므로, 같은 국가 안에서
#   시간적으로 가까운 값으로 결측치를 채우는 것이 전체 평균 대체보다 자연스럽다.
# - 원 팀원 코드는 ffill 후 bfill을 사용했지만, 최종 통합본에서는
#   데이터 누수 위험을 줄이기 위해 모델 파이프라인 안에서 train set 기준
#   median imputation을 사용한다.
# - Country는 순위가 없는 범주형 변수이므로 label encoding이 아니라
#   one-hot encoding을 사용한다. label encoding을 사용하면 모델이
#   국가 사이에 존재하지 않는 대소 관계를 학습할 수 있다.
# - 일반적인 IQR 이상치는 국제 위기 상황에서는 실제 의미 있는 급등/급락일 수 있으므로
#   무조건 제거하지 않는다. 대신 999.0, -99.0처럼 명백한 오류성 값만 먼저 처리한다.
# - scaling은 거리 기반 모델이나 회귀 모델에서 변수 단위 차이의 영향을 줄이기 위해 필요하다.
# - 이 통합본은 사용자의 최종 요청에 맞춰 StandardScaler가 아니라 RobustScaler만 사용한다.
#   RobustScaler는 평균/표준편차 대신 중앙값과 IQR을 사용하므로 극단값 영향에 덜 민감하다.
#
# 4번 역할(모델링 및 성능 평가)에서 강조한 내용:
# - Classification은 Fuel_Price_Change_Percent를 기준으로 유가 상승 정도를
#   No_Increase / Slight_Increase / Large_Increase 세 등급으로 분류한다.
# - Decision Tree는 분류 기준을 tree 구조로 시각화할 수 있어 발표에서 설명하기 좋다.
# - GridSearchCV는 여러 parameter 조합을 비교하여 가장 좋은 Decision Tree 설정을 찾는다.
# - k-fold cross validation은 classification 모델이 특정 train/test split에만
#   우연히 잘 맞는지 확인하고, 과적합을 줄이며 모델 신뢰성을 검증하기 위해 사용한다.
# - classification_report의 precision, recall, F1-score는 accuracy만으로 알 수 없는
#   클래스별 예측 성능을 보여준다.
# - Feature importance는 어떤 변수가 Decision Tree의 분류 결정에 중요했는지 설명한다.
# - Regression은 Fuel_Price_Change_Percent의 연속적인 변동 폭을 예측한다.
# - MSE, RMSE, R2는 회귀 모델의 오차와 설명력을 보여주는 대표 지표이다.
# - SST, SSE, SSR 개념은 회귀에서 실제값, 예측값, 평균 사이의 변동량을 이해하는 데 사용된다.
# - K-Means는 비슷한 시장 상황을 군집화하기 위해 사용한다.
# - inertia는 군집 내부 응집도를, silhouette score는 군집 분리도를 평가한다.
# - elbow method와 silhouette score를 함께 보면 적절한 cluster 개수를 결정하는 근거가 된다.
# - PCA 2D 시각화는 다차원 군집 결과를 발표용 2차원 그림으로 보여주기 위해 사용한다.


DATA_FILE_NAME = "Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv"


# 데이터 파일 후보 경로를 여러 개 둔다.
# 이유:
# - 팀원별 코드는 각자 PC의 절대경로나 현재 폴더 기준 상대경로를 사용했다.
# - 최종 통합본에서는 실행 위치가 바뀌어도 최대한 자동으로 데이터를 찾게 해야 한다.
# - 가장 먼저 이 스크립트 옆의 data/ 폴더를 확인하고, 없으면 현재까지 받은 원본 위치들을 확인한다.
DATA_PATH_CANDIDATES = [
    Path(__file__).resolve().parent / "data" / DATA_FILE_NAME,
    Path(__file__).resolve().parent / DATA_FILE_NAME,
    Path(
        "/Users/winter/Documents/가천대학교/2026년 3학년 1학기/Data Science/Term Project/데이터 셋"
    )
    / DATA_FILE_NAME,
    Path(
        "/Users/winter/Documents/가천대학교/2026년 3학년 1학기/Data Science/Term Project/3번 (조민서)"
    )
    / DATA_FILE_NAME,
]


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_ko"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"


def make_one_hot_encoder():
    """scikit-learn 버전에 따라 OneHotEncoder 인자명이 달라지는 문제를 흡수한다."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def resolve_data_path():
    """후보 경로 중 실제 존재하는 CSV 경로를 반환한다."""
    for path in DATA_PATH_CANDIDATES:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in DATA_PATH_CANDIDATES)
    raise FileNotFoundError(f"Dataset was not found. Tried:\n{tried}")


def prepare_output_dirs():
    """결과 파일을 저장할 폴더를 만든다."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """CSV 파일을 pandas DataFrame으로 읽는다."""
    data_path = resolve_data_path()
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"[LOAD] Data path: {data_path}")
    print(f"[LOAD] Shape: {df.shape}")
    return df


def clean_dirty_values(df):
    """원본 데이터의 오류성 dirty value를 모델링 전에 일관되게 처리한다."""
    cleaned = df.copy()

    # Date는 문자열 그대로 모델에 넣을 수 없으므로 datetime과 ordinal 숫자를 함께 만든다.
    # Date_Ordinal은 날짜 흐름을 수치 feature로 반영하기 위해 사용한다.
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned["Date_Ordinal"] = cleaned["Date"].map(
        lambda value: value.toordinal() if pd.notna(value) else np.nan
    )

    # 같은 국가 안에서 시간 흐름이 유지되도록 정렬한다.
    # 이 정렬은 EDA와 결측치 보정, 날짜별 시각화의 해석 가능성을 높인다.
    cleaned = cleaned.sort_values(["Country", "Date"]).reset_index(drop=True)

    # Fuel_Price_Change_Percent의 999.0은 정상적인 유가 변동률로 보기 어렵다.
    # 실제 정상 최대값은 약 60 근처이므로, 100 초과는 전산 오류성 이상치로 처리한다.
    cleaned.loc[
        cleaned["Fuel_Price_Change_Percent"] > 100,
        "Fuel_Price_Change_Percent",
    ] = np.nan

    # News_Sentiment_Score의 -99.0도 정상 감성 점수 범위를 크게 벗어난다.
    # 모델이 이 값을 의미 있는 매우 부정적 감성으로 오해하지 않도록 NaN으로 바꾼다.
    cleaned.loc[cleaned["News_Sentiment_Score"] == -99.0, "News_Sentiment_Score"] = np.nan

    return cleaned


def run_eda(raw_df, cleaned_df):
    """2번 역할의 EDA 작업을 통합 실행하고, 발표용 표와 그래프를 저장한다."""
    print("[EDA] Saving descriptive tables and figures...")

    # [2번 역할 반영] 원본 데이터의 결측치 현황을 저장한다.
    # 이 표는 전처리 담당자가 어떤 컬럼을 처리해야 하는지 확인하는 근거가 된다.
    raw_df.isna().sum().rename("missing_count").to_csv(METRIC_DIR / "missing_values_raw.csv")

    # [2번 역할 반영] describe() 결과를 저장한다.
    # 평균, 표준편차, 최솟값, 최댓값을 통해 변수별 scale 차이와 이상값 가능성을 파악한다.
    raw_df.describe(include="all").to_csv(METRIC_DIR / "descriptive_statistics_raw.csv")

    # 오류성 dirty value를 NaN으로 바꾼 뒤의 결측치 개수를 저장한다.
    # 이 결과는 999.0, -99.0 처리 후 결측치가 얼마나 늘어났는지 보여준다.
    cleaned_df.isna().sum().rename("missing_count").to_csv(
        METRIC_DIR / "missing_values_after_dirty_value_cleaning.csv"
    )

    # [4번 역할 A안 반영] 3-class classification target의 클래스 분포를 저장한다.
    # 모델 성능 해석에서 클래스 불균형이 있는지 확인하기 위한 표이다.
    target_counts = pd.cut(
        cleaned_df["Fuel_Price_Change_Percent"],
        bins=[-np.inf, 0, 5, np.inf],
        labels=["No_Increase", "Slight_Increase", "Large_Increase"],
        include_lowest=True,
    ).value_counts(dropna=False)
    target_counts.rename("count").to_csv(METRIC_DIR / "classification_target_counts.csv")

    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns

    # 전체 수치형 변수 분포를 확인한다.
    # 이 그래프는 EDA 담당 코드의 histogram 역할을 통합한 것이다.
    cleaned_df[numeric_cols].hist(figsize=(18, 12), bins=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_numeric_histograms.png", dpi=150)
    plt.close()

    # 상관관계 heatmap은 WTI, Brent, OPEC 가격처럼 서로 강하게 연결된 변수를 보여준다.
    corr = cleaned_df[numeric_cols].corr()
    plt.figure(figsize=(14, 10))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=60, ha="right", fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_correlation_heatmap.png", dpi=150)
    plt.close()

    # 국가별 현지 연료 가격 분포를 비교한다.
    # 국가별 화폐 단위가 크게 다르기 때문에 log scale을 사용한다.
    plt.figure(figsize=(14, 7))
    countries = sorted(cleaned_df["Country"].dropna().unique())
    country_data = [
        cleaned_df.loc[cleaned_df["Country"] == country, "Fuel_Price_Local"].dropna()
        for country in countries
    ]
    plt.boxplot(country_data, tick_labels=countries, showfliers=True)
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.title("Fuel Price Distribution by Country")
    plt.xlabel("Country")
    plt.ylabel("Fuel Price Local (Log Scale)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_fuel_price_by_country_boxplot.png", dpi=150)
    plt.close()


def add_classification_target(df):
    """A안 기준 3-class classification target을 만든다."""
    result = df.copy()
    result["Fuel_Rise_Level"] = pd.cut(
        result["Fuel_Price_Change_Percent"],
        bins=[-np.inf, 0, 5, np.inf],
        labels=["No_Increase", "Slight_Increase", "Large_Increase"],
        include_lowest=True,
    )
    return result


def make_feature_matrix(df):
    """모델 입력 X를 만든다. target과 원본 Date는 제거하고, Country는 범주형으로 남긴다."""
    drop_cols = ["Date", "Fuel_Price_Change_Percent", "Fuel_Rise_Level"]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def split_column_types(X):
    """수치형 컬럼과 범주형 컬럼을 분리한다."""
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X):
    """결측치 처리, one-hot encoding, Robust scaling을 하나의 ColumnTransformer로 묶는다."""
    numeric_cols, categorical_cols = split_column_types(X)

    # 수치형 feature 처리:
    # 1. 결측치는 train set의 중앙값으로 대체한다.
    #    원 3번 역할 코드는 시계열 ffill/bfill을 실험했지만,
    #    최종 모델 평가에서는 test set 정보가 train set으로 새지 않도록
    #    Pipeline 안에서 train 기준 imputation을 수행한다.
    # 2. scaling은 사용자가 요청한 대로 RobustScaler만 사용한다.
    #    RobustScaler는 중앙값과 IQR을 기준으로 하므로 오류성 값과 극단 변동값에 덜 민감하다.
    #    국제 위기 상황의 큰 변동값은 연구 대상일 수 있으므로 무조건 삭제하지 않고,
    #    scale 영향을 줄이는 방식으로 다룬다.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # 범주형 feature 처리:
    # 1. 결측치는 최빈값으로 대체한다.
    # 2. Country처럼 순서가 없는 값은 one-hot encoding한다.
    #    국가 이름은 순위형 데이터가 아니므로 label encoding을 사용하면
    #    USA < UK < Germany 같은 잘못된 크기 관계가 생길 수 있다.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    """전처리 이후 feature 이름을 모델 해석용으로 복원한다."""
    feature_names = list(numeric_cols)
    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())
    return feature_names


def run_classification(cleaned_df):
    """4번 역할의 Decision Tree classification을 A안 target 기준으로 통합 실행한다."""
    print("[CLASSIFICATION] Training Decision Tree classifier...")

    # [4번 역할 반영] Fuel_Price_Change_Percent를 구간화하여
    # No_Increase / Slight_Increase / Large_Increase target을 만든다.
    # 원 코드의 A안 target 설정을 유지한다.
    model_df = add_classification_target(cleaned_df)
    model_df = model_df[model_df["Fuel_Rise_Level"].notna()].copy()

    # target 생성에 사용된 Fuel_Price_Change_Percent는 X에서 제거한다.
    # 제거하지 않으면 모델이 정답을 feature로 직접 보게 되는 target leakage가 발생한다.
    X = make_feature_matrix(model_df)
    y = model_df["Fuel_Rise_Level"]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # 사용자가 명시한 4번 역할 조건: shuffle=True.
    # classification에서는 stratify=y를 함께 사용하여 train/test 양쪽의 클래스 비율이
    # 비슷하게 유지되도록 한다. 이렇게 하면 특정 클래스가 test set에 과소/과대 포함되는
    # 상황을 줄일 수 있다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        stratify=y,
        random_state=42,
    )

    classifier = Pipeline(
        steps=[
            # 전처리와 모델을 Pipeline으로 묶는다.
            # 이렇게 하면 cross validation의 각 fold 안에서도 imputer/scaler가
            # train fold에만 fit되어 더 안전하다.
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )

    # Decision Tree parameter 후보들이다.
    # criterion: 분할 품질 측정 방식
    # max_depth: tree 깊이 제한
    # min_samples_split: 내부 노드를 분할하기 위한 최소 sample 수
    # min_samples_leaf: leaf node에 있어야 하는 최소 sample 수
    # class_weight: 클래스 불균형 보정 여부
    param_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [3, 4, 5, 6, None],
        "model__min_samples_split": [5, 10, 20, 30],
        "model__min_samples_leaf": [5, 10, 20, 30],
        "model__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        classifier,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        # n_jobs=-1 can fail in restricted execution environments because joblib
        # tries to create multiple worker processes. n_jobs=1 is slower but stable.
        n_jobs=1,
    )
    # GridSearchCV는 위 parameter 조합을 5-fold로 비교해 f1_macro 기준 최적 모델을 찾는다.
    # f1_macro는 클래스별 F1을 평균내므로, 특정 클래스만 잘 맞히는 모델보다
    # 세 클래스를 고르게 맞히는 모델을 선호한다.
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # 과제 지침의 필수 조건인 k-fold cross validation 결과를 별도로 저장한다.
    # 여기서는 최종 선택된 모델을 train set에서 다시 5-fold accuracy로 평가한다.
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
    )

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(METRIC_DIR / "classification_report.csv")

    metrics = {
        "best_params": grid_search.best_params_,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "cv_accuracy_scores": [float(score) for score in cv_scores],
        "cv_accuracy_mean": float(np.mean(cv_scores)),
    }
    (METRIC_DIR / "classification_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation=30)
    plt.title("Decision Tree Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "classification_confusion_matrix.png", dpi=150)
    plt.close()

    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_tree = best_model.named_steps["model"]
    feature_names = get_feature_names(fitted_preprocessor, numeric_cols, categorical_cols)

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": fitted_tree.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    importance_df.to_csv(METRIC_DIR / "classification_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    top10 = importance_df.head(10).iloc[::-1]
    plt.barh(top10["Feature"], top10["Importance"])
    plt.title("Top 10 Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "classification_feature_importance.png", dpi=150)
    plt.close()

    plt.figure(figsize=(30, 15), dpi=120)
    plot_tree(
        fitted_tree,
        feature_names=feature_names,
        class_names=[str(value) for value in fitted_tree.classes_],
        filled=True,
        rounded=True,
        fontsize=6,
        max_depth=4,
        impurity=False,
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "classification_decision_tree_structure.png", dpi=150)
    plt.close()

    print(f"[CLASSIFICATION] Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"[CLASSIFICATION] 5-fold CV mean accuracy: {metrics['cv_accuracy_mean']:.4f}")
    return metrics


def run_regression(cleaned_df):
    """Fuel_Price_Change_Percent의 연속적인 변동률을 예측하는 회귀 모델을 실행한다."""
    print("[REGRESSION] Training Ridge regression model...")

    # 회귀 target도 Fuel_Price_Change_Percent이다.
    # 이전 팀원 수정본에서 target이 Fuel_Price_Local이 아니라
    # Fuel_Price_Change_Percent로 바뀐 내용을 반영했다.
    # clean_dirty_values()에서 999.0 오류값은 이미 NaN으로 바뀌었으므로,
    # target이 NaN인 행은 학습에서 제외한다.
    model_df = cleaned_df[cleaned_df["Fuel_Price_Change_Percent"].notna()].copy()

    X = make_feature_matrix(model_df)
    y = model_df["Fuel_Price_Change_Percent"]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # 사용자가 요청한 최종 기준에 맞춰 regression에서도 shuffle=True를 사용한다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    regression_model = Pipeline(
        steps=[
            # classification과 동일하게 전처리와 모델을 Pipeline으로 묶어
            # train/test 분리 이후의 fit/transform 순서를 안전하게 유지한다.
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=3.0)),
        ]
    )
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)

    # MSE: 실제값과 예측값 차이의 제곱 평균이다.
    # RMSE: MSE의 제곱근이며 target과 같은 단위라 해석이 더 쉽다.
    # R2: 모델이 target 변동을 얼마나 설명하는지 나타낸다.
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "model": "Ridge(alpha=3.0)",
        "target": "Fuel_Price_Change_Percent",
        "test_mse": float(mse),
        "test_rmse": rmse,
        "test_r2": float(r2)
    }
    (METRIC_DIR / "regression_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    comparison.to_csv(METRIC_DIR / "regression_actual_vs_predicted.csv", index=False)

    # 회귀 계수는 각 feature가 target 증가/감소 방향에 어떻게 연결되는지 보여준다.
    # 단, 모든 feature가 Robust scaling된 뒤의 계수이므로 원래 단위 그대로 해석하면 안 된다.
    fitted_preprocessor = regression_model.named_steps["preprocessor"]
    fitted_regressor = regression_model.named_steps["model"]
    feature_names = get_feature_names(fitted_preprocessor, numeric_cols, categorical_cols)
    coefficient_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": fitted_regressor.coef_,
        }
    ).sort_values("Coefficient", ascending=False)
    coefficient_df.to_csv(METRIC_DIR / "regression_coefficients.csv", index=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    slope, intercept = np.polyfit(y_test, y_pred, 1)
    x_line = np.linspace(min_value, max_value, 200)
    y_line = slope * x_line + intercept
    plt.plot( x_line, y_line, label="Ridge Trend Line")
    plt.title("Actual vs Predicted Fuel Price Change Percent")
    plt.xlabel("Actual Fuel_Price_Change_Percent")
    plt.ylabel("Predicted Fuel_Price_Change_Percent")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "regression_actual_vs_predicted.png", dpi=150)
    plt.close()

    print(f"[REGRESSION] RMSE: {rmse:.4f}")
    print(f"[REGRESSION] R2: {r2:.4f}")
    return metrics


def run_clustering(cleaned_df):
    """K-Means clustering을 보조 분석으로 실행한다."""
    print("[CLUSTERING] Training K-Means clustering models...")

    # [4번 역할 K-Means 반영] 군집화에는 위기 상황을 설명하는 핵심 변수만 사용한다.
    # Brent: 국제 유가 benchmark
    # Fuel_Price_Local: 국가별 현지 연료 가격
    # Fuel_Price_Change_Percent: 가격 변동률
    # Crisis_Intensity_Index: 지정학적 위기 강도
    # USD_Exchange_Rate: 환율 영향
    # Inflation_Rate_Percent: 거시경제 물가 압력
    clustering_cols = [
        "Brent_Crude_USD_per_barrel",
        "Fuel_Price_Local",
        "Fuel_Price_Change_Percent",
        "Crisis_Intensity_Index",
        "USD_Exchange_Rate",
        "Inflation_Rate_Percent",
    ]

    cluster_df = cleaned_df[["Date", "Country"] + clustering_cols].copy()
    X = cluster_df[clustering_cols]

    # K-Means 역시 거리 기반 알고리즘이므로 scaling의 영향을 크게 받는다.
    # 변수 단위가 다르면 큰 단위의 변수가 거리 계산을 지배할 수 있다.
    # 사용자가 요청한 대로 RobustScaler만 사용한다.
    cluster_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    X_scaled = cluster_preprocessor.fit_transform(X)

    k_results = []
    max_k = len(clustering_cols) * 2
    # 여러 k 값을 비교한다.
    # inertia는 cluster 내부 거리 합이므로 낮을수록 응집도가 좋다.
    # silhouette score는 군집 내부 응집도와 군집 간 분리도를 함께 보며,
    # 1에 가까울수록 군집이 잘 분리되었다고 해석한다.
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        k_results.append(
            {
                "n_clusters": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(X_scaled, labels)),
            }
        )

    k_results_df = pd.DataFrame(k_results)
    k_results_df.to_csv(METRIC_DIR / "clustering_k_evaluation.csv", index=False)
    # 원 팀원 코드에서는 best_k를 8로 고정했지만,
    # 최종 통합본에서는 silhouette score가 가장 높은 k를 자동 선택한다.
    best_k = int(k_results_df.loc[k_results_df["silhouette_score"].idxmax(), "n_clusters"])

    best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_df["Cluster"] = best_kmeans.fit_predict(X_scaled)

    cluster_counts = cluster_df["Cluster"].value_counts().sort_index()
    cluster_counts.rename("count").to_csv(METRIC_DIR / "clustering_counts.csv")

    cluster_centers = pd.DataFrame(best_kmeans.cluster_centers_, columns=clustering_cols)
    cluster_centers.to_csv(METRIC_DIR / "clustering_centers_scaled.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(k_results_df["n_clusters"], k_results_df["inertia"], marker="o")
    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "clustering_elbow.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_results_df["n_clusters"], k_results_df["silhouette_score"], marker="o")
    plt.title("K-Means Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "clustering_silhouette.png", dpi=150)
    plt.close()

    # PCA는 여러 feature로 계산된 cluster 결과를 2차원 평면에 시각화하기 위한 도구이다.
    # K-Means 자체는 원래 feature 공간에서 수행되고, PCA는 발표용 시각화에만 사용한다.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(best_kmeans.cluster_centers_)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_df["Cluster"], alpha=0.7)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c="black", marker="o", s=30, label="Cluster Centers")
    plt.title(f"K-Means Clustering PCA View (k={best_k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "clustering_pca.png", dpi=150)
    plt.close()

    metrics = {
        "best_k": best_k,
        "best_silhouette_score": float(
            k_results_df.loc[k_results_df["n_clusters"] == best_k, "silhouette_score"].iloc[0]
        ),
    }
    (METRIC_DIR / "clustering_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"[CLUSTERING] Best k: {best_k}")
    print(f"[CLUSTERING] Best silhouette score: {metrics['best_silhouette_score']:.4f}")

    plot_df = cluster_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")

    # Country는 문자형 categorical 데이터이므로 y축에 표시하기 위해 숫자 코드로 변환
    countries = sorted(plot_df["Country"].unique())
    plot_df["Country_Code"] = pd.Categorical(
        plot_df["Country"],
        categories=countries,
        ordered=True
    ).codes

    cluster_labels = sorted(plot_df["Cluster"].unique())
    n_clusters = len(cluster_labels)

    # tab10에서 실제 클러스터 개수만큼만 색상 사용
    base_cmap = plt.get_cmap("tab10")
    colors = base_cmap(range(n_clusters))
    cmap = ListedColormap(colors)

    # 클러스터 라벨을 0부터 연속된 코드로 변환
    cluster_to_code = {label: i for i, label in enumerate(cluster_labels)}
    plot_df["Cluster_Code"] = plot_df["Cluster"].map(cluster_to_code)

    bounds = range(n_clusters + 1)
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        plot_df["Date"],
        plot_df["Country_Code"],
        c=plot_df["Cluster_Code"],
        cmap=cmap,
        norm=norm,
        alpha=0.75,
        s=35
    )

    plt.title(f"K-Means Clustering by Date and Country (k={n_clusters})")
    plt.xlabel("Date")
    plt.ylabel("Country")

    plt.yticks(range(len(countries)), countries)
    plt.xticks(rotation=45)

    cbar = plt.colorbar(scatter, ticks=range(n_clusters))
    cbar.set_label("Cluster")
    cbar.set_ticklabels(cluster_labels)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "clustering_date_country_scatter.png", dpi=150)
    plt.close()

    return metrics


def validate_outputs():
    """실행 결과 파일이 실제로 생성되었는지 확인한다."""
    required_files = [
        METRIC_DIR / "classification_metrics.json",
        METRIC_DIR / "classification_report.csv",
        METRIC_DIR / "regression_metrics.json",
        METRIC_DIR / "clustering_metrics.json",
        FIGURE_DIR / "classification_confusion_matrix.png",
        FIGURE_DIR / "regression_actual_vs_predicted.png",
        FIGURE_DIR / "clustering_pca.png",
    ]
    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise RuntimeError(f"Some required output files were not created:\n{missing_text}")
    print("[VALIDATION] Required output files were created successfully.")


def main():
    prepare_output_dirs()

    raw_df = load_data()
    cleaned_df = clean_dirty_values(raw_df)

    run_eda(raw_df, cleaned_df)
    classification_metrics = run_classification(cleaned_df)
    regression_metrics = run_regression(cleaned_df)
    clustering_metrics = run_clustering(cleaned_df)

    validate_outputs()

    summary = {
        "classification": classification_metrics,
        "regression": regression_metrics,
        "clustering": clustering_metrics,
    }
    (OUTPUT_DIR / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("[DONE] Integrated pipeline finished successfully.")
    print(f"[DONE] Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
