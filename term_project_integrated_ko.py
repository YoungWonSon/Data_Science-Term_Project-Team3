"""
Team 3 Data Science Term Project - Integrated Pipeline (Korean comments)

기존 팀원 코드 기반 통합본:
- Seo_Jangwon_EDA.py: 데이터 구조, 결측치, 오류성 이상치, 분포, 상관관계 분석
- Jo_Minseo_*.py: 결측치 처리, 오류성 이상치 처리, one-hot encoding, scaling
- Classification_Decison_Tree.py: Decision Tree 분류, GridSearchCV, k-fold CV, feature importance
- KIm_Dana_Multi-Linear-Regression.py: 다중 선형 회귀, MSE/RMSE/R2, 실제값-예측값 비교
- KIm_Dana_K-Mean-Clustering.py: K-Means, silhouette score, elbow/PCA 시각화
- Kwon_Keonho_Regression_OpenSource.py: top-level 함수 구조와 결과 저장 방식

최종 목표:
1. Classification: 다음날 현지 연료 가격이 상승할지 여부를 이진 분류
2. Regression: 다음날 현지 연료 가격 변동폭을 예측
3. Clustering: 위기/유가/환율/인플레이션 기반 시장 패턴 보조 분석
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


warnings.filterwarnings("ignore")


DEFAULT_CSV_PATH = (
    "/Users/winter/Documents/가천대학교/2026년 3학년 1학기/"
    "Data Science/Term Project/데이터 셋/Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv"
)


TEAM_CODE_REFERENCES = {
    "eda": "Seo_Jangwon_EDA.py",
    "missing_values": "Jo_Minseo_fillna.py",
    "one_hot_encoding": "Jo_Minseo_Onehotencoding.py",
    "outlier_detection": "Jo_Minseo_Outlier.py",
    "outlier_delete_test": "Jo_Minseo_Outlierdelete.py",
    "outlier_replace": "Jo_Minseo_Outlierreplace.py",
    "scaling": "Jo_Minseo_Scaling.py, Kim_Dana_Standarization_Data.py",
    "classification": "Classification_Decison_Tree.py",
    "regression": "KIm_Dana_Multi-Linear-Regression.py",
    "clustering": "KIm_Dana_K-Mean-Clustering.py",
}


def make_one_hot_encoder() -> OneHotEncoder:
    """scikit-learn 버전에 따라 sparse_output 인자가 없을 수 있어 호환 처리한다."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_output_dirs(output_dir: str | Path) -> Dict[str, Path]:
    """EDA, model, clustering 결과를 저장할 폴더를 생성한다."""
    base = Path(output_dir)
    dirs = {
        "base": base,
        "eda": base / "eda",
        "classification": base / "classification",
        "regression": base / "regression",
        "clustering": base / "clustering",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """기존 팀원 코드처럼 read_csv로 원본 dirty dataset을 불러온다."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "Date" not in df.columns or "Country" not in df.columns:
        raise ValueError("Dataset must contain 'Date' and 'Country' columns.")
    return df


def run_eda(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    2번 역할 코드 기반 EDA.

    기존 코드는 print와 plt.show 중심이었으므로,
    최종 제출용으로 표와 그래프를 파일로 저장하도록 수정했다.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shape_df = pd.DataFrame({"rows": [df.shape[0]], "columns": [df.shape[1]]})
    dtypes_df = pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]})
    describe_df = df.describe(include="all").T
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["column", "missing_count"]
    missing_df["missing_percent"] = missing_df["missing_count"] / len(df) * 100

    error_values_df = pd.DataFrame(
        {
            "column": df.columns,
            "count_999": [(df[c] == 999.0).sum() if pd.api.types.is_numeric_dtype(df[c]) else 0 for c in df.columns],
            "count_minus_99": [
                (df[c] == -99.0).sum() if pd.api.types.is_numeric_dtype(df[c]) else 0 for c in df.columns
            ],
        }
    )

    corr_matrix = df.corr(numeric_only=True)
    corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs < 1].drop_duplicates().head(20).reset_index()
    corr_pairs.columns = ["feature_1", "feature_2", "abs_correlation"]

    shape_df.to_csv(out / "dataset_shape.csv", index=False, encoding="utf-8-sig")
    dtypes_df.to_csv(out / "dataset_dtypes.csv", index=False, encoding="utf-8-sig")
    describe_df.to_csv(out / "descriptive_statistics.csv", encoding="utf-8-sig")
    missing_df.to_csv(out / "missing_values.csv", index=False, encoding="utf-8-sig")
    error_values_df.to_csv(out / "dirty_error_values.csv", index=False, encoding="utf-8-sig")
    corr_pairs.to_csv(out / "top_correlations.csv", index=False, encoding="utf-8-sig")

    # 기존 EDA의 histogram 시각화
    df.hist(figsize=(16, 11))
    plt.tight_layout()
    plt.savefig(out / "numeric_histograms.png", dpi=300)
    plt.close()

    # Fuel_Price_Change_Percent의 999 오류값은 제외하고 박스플롯을 그린다.
    if "Fuel_Price_Change_Percent" in df.columns:
        filtered_data = df[df["Fuel_Price_Change_Percent"] < 100]
        plt.figure(figsize=(8, 4))
        sns.boxplot(
            x=filtered_data["Fuel_Price_Change_Percent"],
            flierprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markeredgecolor": "red",
                "markersize": 4,
            },
        )
        plt.title("Fuel Price Change Percent Boxplot")
        plt.xlabel("Fuel Price Change Percent")
        plt.tight_layout()
        plt.savefig(out / "fuel_price_change_boxplot.png", dpi=300)
        plt.close()

    # 변수 간 상관관계 heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png", dpi=300)
    plt.close()

    # 국가별 현지 연료 가격 분포. 국가별 통화 단위 차이가 커서 로그 스케일을 유지한다.
    if {"Country", "Fuel_Price_Local"}.issubset(df.columns):
        plt.figure(figsize=(14, 7))
        sns.boxplot(
            x="Country",
            y="Fuel_Price_Local",
            data=df,
            flierprops={
                "marker": "o",
                "markerfacecolor": "red",
                "markeredgecolor": "red",
                "markersize": 4,
            },
        )
        plt.yscale("log")
        plt.title("Fuel Price Distribution by Country")
        plt.xlabel("Country")
        plt.ylabel("Fuel Price Local (Log Scale)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out / "fuel_price_by_country_boxplot.png", dpi=300)
        plt.close()

    return {
        "shape": shape_df,
        "dtypes": dtypes_df,
        "describe": describe_df,
        "missing": missing_df,
        "error_values": error_values_df,
        "top_correlations": corr_pairs,
    }


def clean_dirty_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    3번 역할의 Outlierreplace.py 기반 처리.

    999.0과 -99.0은 실제 경제적 극단값이라기보다 입력 오류성 값으로 보고 NaN으로 변환한다.
    IQR 기반 이상치는 위기 상황의 의미 있는 신호일 수 있으므로 일괄 삭제하지 않는다.
    """
    df = df.copy()

    if "Fuel_Price_Change_Percent" in df.columns:
        df["Fuel_Price_Change_Percent"] = df["Fuel_Price_Change_Percent"].replace(999.0, np.nan)

    if "News_Sentiment_Score" in df.columns:
        df["News_Sentiment_Score"] = df["News_Sentiment_Score"].replace(-99.0, np.nan)

    return df


def fill_time_series_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    3번 역할의 fillna.py 기반 결측치 처리.

    원본 코드는 국가별 ffill 후 bfill을 사용했다.
    최종 예측 통합본에서는 미래값 사용 위험을 줄이기 위해 ffill을 먼저 쓰고,
    앞쪽 결측치는 국가별 중앙값과 전체 중앙값으로 보완한다.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df.groupby("Country")[col].transform(lambda s: s.ffill())
        df[col] = df.groupby("Country")[col].transform(lambda s: s.fillna(s.median()))
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df


def add_next_day_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    제안서 목표에 맞게 다음날 target을 생성한다.

    Classification target:
        Next_Day_Fuel_Rise = 다음날 Fuel_Price_Change_Percent가 0보다 크면 1, 아니면 0

    Regression target:
        Next_Day_Fuel_Change_Amount = 다음날 Fuel_Price_Local - 오늘 Fuel_Price_Local
    """
    df = df.copy()
    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)

    df["Next_Day_Fuel_Price_Local"] = df.groupby("Country")["Fuel_Price_Local"].shift(-1)
    df["Next_Day_Fuel_Change_Percent"] = df.groupby("Country")["Fuel_Price_Change_Percent"].shift(-1)
    df["Next_Day_Fuel_Change_Amount"] = df["Next_Day_Fuel_Price_Local"] - df["Fuel_Price_Local"]
    df["Next_Day_Fuel_Rise"] = (df["Next_Day_Fuel_Change_Percent"] > 0).astype("Int64")

    target_cols = ["Next_Day_Fuel_Price_Local", "Next_Day_Fuel_Change_Percent", "Next_Day_Fuel_Change_Amount", "Next_Day_Fuel_Rise"]
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    df["Next_Day_Fuel_Rise"] = df["Next_Day_Fuel_Rise"].astype(int)

    return df


def prepare_modeling_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    모델링용 X, classification y, regression y를 만든다.

    4번 코드의 Date_Ordinal 아이디어를 유지하되,
    미래 target 컬럼은 feature에서 제외한다.
    """
    df = df.copy()
    df["Date_Ordinal"] = df["Date"].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    drop_cols = [
        "Next_Day_Fuel_Rise",
        "Next_Day_Fuel_Change_Amount",
        "Next_Day_Fuel_Price_Local",
        "Next_Day_Fuel_Change_Percent",
    ]
    X = df.drop(columns=drop_cols, errors="ignore")
    y_classification = df["Next_Day_Fuel_Rise"]
    y_regression = df["Next_Day_Fuel_Change_Amount"]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    return X, y_classification, y_regression, numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler_name: str = "standard",
    imputer_strategy: str = "median",
) -> ColumnTransformer:
    """
    3번의 인코딩/스케일링과 4번의 SimpleImputer 사용을 하나로 묶는다.

    train 데이터에만 fit되고 test 데이터에는 transform만 되도록 Pipeline 내부에 넣는다.
    """
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_name must be 'standard' or 'robust'.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", scaler),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


def time_based_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    시계열 성격을 고려해 날짜 순서 기준으로 train/test를 나눈다.

    기존 4번 classification은 stratify random split을 사용했지만,
    최종 next-day 예측에서는 미래 데이터가 train에 섞이지 않는 편이 더 안전하다.
    """
    ordered_index = X.sort_values(["Date", "Country"]).index
    split_point = int(len(ordered_index) * (1 - test_size))
    train_index = ordered_index[:split_point]
    test_index = ordered_index[split_point:]

    return X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]


def run_classification(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    output_dir: str | Path,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """4번 Decision Tree classification 코드를 next-day binary target에 맞게 수정한 모델링 함수."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_size=test_size)

    # 날짜 컬럼은 Date_Ordinal로 이미 반영했으므로 원본 Date는 제거한다.
    X_train = X_train.drop(columns=["Date"], errors="ignore")
    X_test = X_test.drop(columns=["Date"], errors="ignore")
    categorical_cols = [c for c in categorical_cols if c != "Date"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_name="standard", imputer_strategy="median")
    classifier = DecisionTreeClassifier(random_state=random_state)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])

    param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 4, 5, 6, None],
        "classifier__min_samples_split": [5, 10, 20, 30],
        "classifier__min_samples_leaf": [5, 10, 20, 30],
        "classifier__class_weight": [None, "balanced"],
    }

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=1,
        return_train_score=False,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=["actual_0", "actual_1"], columns=["pred_0", "pred_1"])
    cv_results_df = pd.DataFrame(grid_search.cv_results_).sort_values("rank_test_score")

    pd.DataFrame([metrics]).to_csv(out / "classification_test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report_dict).T.to_csv(out / "classification_report.csv", encoding="utf-8-sig")
    confusion_df.to_csv(out / "classification_confusion_matrix.csv", encoding="utf-8-sig")
    cv_results_df.to_csv(out / "classification_gridsearch_results.csv", index=False, encoding="utf-8-sig")

    predictions_df = pd.DataFrame(
        {
            "Actual_Next_Day_Fuel_Rise": y_test.values,
            "Predicted_Next_Day_Fuel_Rise": y_pred,
        },
        index=y_test.index,
    )
    predictions_df.to_csv(out / "classification_predictions.csv", encoding="utf-8-sig")

    feature_importance_df = extract_tree_feature_importance(best_model)
    feature_importance_df.to_csv(out / "classification_feature_importance.csv", index=False, encoding="utf-8-sig")

    if not feature_importance_df.empty:
        top10 = feature_importance_df.head(10)
        plt.figure(figsize=(10, 6))
        plt.barh(top10["feature"], top10["importance"])
        plt.gca().invert_yaxis()
        plt.title("Top 10 Feature Importances - Decision Tree")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(out / "classification_feature_importance_top10.png", dpi=300)
        plt.close()

    # 4번 코드의 tree plot 아이디어를 유지하되 파일로 저장한다.
    try:
        tree_model = best_model.named_steps["classifier"]
        feature_names = best_model.named_steps["preprocess"].get_feature_names_out()
        plt.figure(figsize=(30, 15), dpi=150)
        plot_tree(
            tree_model,
            feature_names=list(feature_names),
            class_names=["No Rise", "Rise"],
            filled=True,
            rounded=True,
            fontsize=6,
            max_depth=5,
            impurity=False,
        )
        plt.title("Decision Tree Structure")
        plt.tight_layout()
        plt.savefig(out / "classification_decision_tree.png", dpi=300)
        plt.close()
    except Exception:
        pass

    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": confusion_df,
        "feature_importance": feature_importance_df,
    }


def extract_tree_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Decision Tree의 feature importance를 전처리 후 feature 이름과 함께 추출한다."""
    try:
        preprocessor = model.named_steps["preprocess"]
        classifier = model.named_steps["classifier"]
        feature_names = preprocessor.get_feature_names_out()
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": classifier.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """4번 회귀 코드의 MSE/RMSE/R2에 MAE를 추가한다."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def run_regression(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    output_dir: str | Path,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """4번 Linear Regression 코드를 다음날 가격 변동폭 예측으로 수정한 회귀 함수."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_size=test_size)
    X_train = X_train.drop(columns=["Date"], errors="ignore")
    X_test = X_test.drop(columns=["Date"], errors="ignore")
    categorical_cols = [c for c in categorical_cols if c != "Date"]

    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    result_rows = []
    fitted_entries = []
    for imputer_strategy in ["median", "mean"]:
        for scaler_name in ["standard", "robust"]:
            preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_name=scaler_name, imputer_strategy=imputer_strategy)
            model = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("regressor", LinearRegression()),
                ]
            )

            cv_result = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_metrics = evaluate_regression(y_test, y_pred)

            row = {
                "model_name": "Multiple Linear Regression",
                "imputer_strategy": imputer_strategy,
                "scaler_name": scaler_name,
                "CV_MAE": -cv_result["test_MAE"].mean(),
                "CV_RMSE": -cv_result["test_RMSE"].mean(),
                "CV_R2": cv_result["test_R2"].mean(),
                "Test_MAE": test_metrics["MAE"],
                "Test_MSE": test_metrics["MSE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_R2": test_metrics["R2"],
            }
            result_rows.append(row)
            fitted_entries.append({"row": row, "model": model, "y_pred": y_pred})

    results_df = pd.DataFrame(result_rows).sort_values(["CV_RMSE", "Test_RMSE"]).reset_index(drop=True)
    best_params = results_df.iloc[0].to_dict()
    best_entry = next(
        entry
        for entry in fitted_entries
        if entry["row"]["imputer_strategy"] == best_params["imputer_strategy"]
        and entry["row"]["scaler_name"] == best_params["scaler_name"]
    )
    best_model = best_entry["model"]
    y_pred = best_entry["y_pred"]
    final_metrics = evaluate_regression(y_test, y_pred)

    results_df.to_csv(out / "regression_model_results.csv", index=False, encoding="utf-8-sig")
    results_df.head(5).to_csv(out / "regression_top5_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([final_metrics]).to_csv(out / "regression_test_metrics.csv", index=False, encoding="utf-8-sig")

    predictions_df = pd.DataFrame(
        {
            "Actual_Next_Day_Fuel_Change_Amount": y_test.values,
            "Predicted_Next_Day_Fuel_Change_Amount": y_pred,
            "Residual": y_test.values - y_pred,
        },
        index=y_test.index,
    )
    predictions_df.to_csv(out / "regression_predictions.csv", encoding="utf-8-sig")

    coefficients_df = extract_regression_coefficients(best_model)
    coefficients_df.to_csv(out / "regression_coefficients_top30.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_value = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--", label="Perfect Prediction Line")
    trend_coef = np.polyfit(y_test, y_pred, 1)
    trend_line = np.poly1d(trend_coef)
    x_line = np.linspace(float(np.min(y_test)), float(np.max(y_test)), 100)
    plt.plot(x_line, trend_line(x_line), linewidth=2, label="Prediction Trend Line")
    plt.title("Actual vs Predicted: Next-Day Fuel Change Amount")
    plt.xlabel("Actual Change Amount")
    plt.ylabel("Predicted Change Amount")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "regression_actual_vs_predicted.png", dpi=300)
    plt.close()

    return {
        "best_model": best_model,
        "best_params": best_params,
        "metrics": final_metrics,
        "all_results": results_df,
        "top5_results": results_df.head(5),
        "coefficients": coefficients_df,
    }


def extract_regression_coefficients(model: Pipeline, top_n: int = 30) -> pd.DataFrame:
    """선형 회귀 계수를 feature 이름과 함께 추출한다."""
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        coefficients = model.named_steps["regressor"].coef_
        coef_df = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "abs_coefficient": np.abs(coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False)
        return coef_df.head(top_n)
    except Exception:
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])


def run_clustering(
    df: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    4번 K-Means clustering 코드를 보조 분석으로 정리한다.

    기존 코드는 best_k=8을 하드코딩했지만,
    최종 통합본은 silhouette score 기준으로 best_k를 자동 선택한다.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clustering_cols = [
        "Brent_Crude_USD_per_barrel",
        "Fuel_Price_Local",
        "Fuel_Price_Change_Percent",
        "Crisis_Intensity_Index",
        "USD_Exchange_Rate",
        "Inflation_Rate_Percent",
    ]
    available_cols = [col for col in clustering_cols if col in df.columns]
    X = df[available_cols].select_dtypes(include=[np.number]).dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_results = []
    max_k = min(12, max(3, len(available_cols) * 2))
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        k_results.append(
            {
                "n_clusters": k,
                "inertia": kmeans.inertia_,
                "silhouette_score": silhouette_score(X_scaled, labels),
            }
        )

    k_results_df = pd.DataFrame(k_results).sort_values("silhouette_score", ascending=False)
    best_k = int(k_results_df.iloc[0]["n_clusters"])
    best_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    cluster_labels = best_kmeans.fit_predict(X_scaled)

    clustered_df = df.loc[X.index].copy()
    clustered_df["Cluster"] = cluster_labels
    cluster_centers_df = pd.DataFrame(
        scaler.inverse_transform(best_kmeans.cluster_centers_),
        columns=available_cols,
    )

    k_results_df.to_csv(out / "kmeans_k_evaluation.csv", index=False, encoding="utf-8-sig")
    clustered_df.to_csv(out / "kmeans_clustered_data.csv", index=False, encoding="utf-8-sig")
    cluster_centers_df.to_csv(out / "kmeans_cluster_centers_original_scale.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.plot(k_results_df.sort_values("n_clusters")["n_clusters"], k_results_df.sort_values("n_clusters")["inertia"], marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "kmeans_elbow_method.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        k_results_df.sort_values("n_clusters")["n_clusters"],
        k_results_df.sort_values("n_clusters")["silhouette_score"],
        marker="o",
    )
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "kmeans_silhouette_scores.png", dpi=300)
    plt.close()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(best_kmeans.cluster_centers_)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, alpha=0.7)
    plt.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        c=np.arange(best_k),
        cmap="viridis",
        marker=".",
        s=300,
        edgecolors="black",
        linewidths=1.5,
    )
    plt.title(f"K-Means Clustering with PCA 2D Visualization, k={best_k}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "kmeans_pca_visualization.png", dpi=300)
    plt.close()

    return {
        "best_k": best_k,
        "k_results": k_results_df,
        "cluster_centers": cluster_centers_df,
        "clustered_data": clustered_df,
    }


def save_summary_json(summary: Dict[str, Any], output_path: str | Path) -> None:
    """모델 핵심 결과를 JSON으로 저장한다."""
    serializable = {}
    for key, value in summary.items():
        if isinstance(value, pd.DataFrame):
            serializable[key] = value.to_dict(orient="records")
        elif isinstance(value, (np.integer, np.floating)):
            serializable[key] = value.item()
        elif isinstance(value, dict):
            serializable[key] = {
                k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in value.items()
                if not hasattr(v, "predict")
            }
        else:
            serializable[key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def run_integrated_pipeline(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    output_dir: str | Path = "term_project_integrated_outputs_ko",
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
    run_cluster: bool = True,
) -> Dict[str, Any]:
    """
    최종 통합 top-level function.

    과제 지침의 end-to-end process에 맞춰
    EDA -> preprocessing -> classification -> regression -> optional clustering -> result saving
    순서로 실행한다.
    """
    dirs = ensure_output_dirs(output_dir)

    raw_df = load_dataset(csv_path)
    eda_results = run_eda(raw_df, dirs["eda"])

    cleaned_df = clean_dirty_values(raw_df)
    filled_df = fill_time_series_missing_values(cleaned_df)
    modeling_df = add_next_day_targets(filled_df)
    modeling_df.to_csv(dirs["base"] / "preprocessed_modeling_dataset.csv", index=False, encoding="utf-8-sig")

    X, y_classification, y_regression, numeric_cols, categorical_cols = prepare_modeling_data(modeling_df)

    classification_results = run_classification(
        X,
        y_classification,
        numeric_cols,
        categorical_cols,
        dirs["classification"],
        cv=cv,
        test_size=test_size,
        random_state=random_state,
    )
    regression_results = run_regression(
        X,
        y_regression,
        numeric_cols,
        categorical_cols,
        dirs["regression"],
        cv=cv,
        test_size=test_size,
        random_state=random_state,
    )
    clustering_results = run_clustering(modeling_df, dirs["clustering"], random_state=random_state) if run_cluster else None

    summary = {
        "team_code_references": TEAM_CODE_REFERENCES,
        "raw_shape": {"rows": raw_df.shape[0], "columns": raw_df.shape[1]},
        "modeling_shape": {"rows": modeling_df.shape[0], "columns": modeling_df.shape[1]},
        "classification_best_params": classification_results["best_params"],
        "classification_metrics": classification_results["metrics"],
        "regression_best_params": regression_results["best_params"],
        "regression_metrics": regression_results["metrics"],
        "clustering_best_k": clustering_results["best_k"] if clustering_results else None,
    }
    save_summary_json(summary, dirs["base"] / "integrated_summary.json")

    return {
        "summary": summary,
        "eda": eda_results,
        "classification": classification_results,
        "regression": regression_results,
        "clustering": clustering_results,
        "output_dir": str(dirs["base"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Team 3 integrated data science term project pipeline.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Path to Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv")
    parser.add_argument("--output-dir", default="term_project_integrated_outputs_ko", help="Directory to save outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test data ratio")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-cluster", action="store_true", help="Skip K-Means clustering analysis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_integrated_pipeline(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        cv=args.cv,
        random_state=args.random_state,
        run_cluster=not args.skip_cluster,
    )

    print("Integrated pipeline completed.")
    print(f"Output directory: {result['output_dir']}")
    print("\nClassification metrics:")
    print(result["summary"]["classification_metrics"])
    print("\nRegression metrics:")
    print(result["summary"]["regression_metrics"])
    if result["summary"]["clustering_best_k"] is not None:
        print(f"\nBest K for clustering: {result['summary']['clustering_best_k']}")
