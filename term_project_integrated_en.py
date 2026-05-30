"""
Team 3 Data Science Term Project - Integrated Pipeline (English comments)

This file is the English-comment version of the final integrated code.
It is based on the team's submitted files:
- Seo_Jangwon_EDA.py: data structure, missing values, dirty values, distribution, correlation
- Jo_Minseo_*.py: missing value handling, dirty value replacement, one-hot encoding, scaling
- Classification_Decison_Tree.py: Decision Tree classification, GridSearchCV, k-fold CV
- KIm_Dana_Multi-Linear-Regression.py: multiple linear regression and regression metrics
- KIm_Dana_K-Mean-Clustering.py: K-Means, silhouette score, elbow method, PCA visualization
- Kwon_Keonho_Regression_OpenSource.py: reusable top-level pipeline and output saving style

Final objectives:
1. Classification: predict whether local fuel price rises on the next day.
2. Regression: predict the next-day local fuel price change amount.
3. Clustering: perform supporting market-pattern analysis using fuel, crisis, FX, and inflation features.
"""

from __future__ import annotations

import argparse
import json
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
    """Create a OneHotEncoder that works with both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_output_dirs(output_dir: str | Path) -> Dict[str, Path]:
    """Create output folders for EDA, classification, regression, and clustering."""
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
    """Load the original dirty CSV dataset using pandas, as in the submitted team code."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "Date" not in df.columns or "Country" not in df.columns:
        raise ValueError("Dataset must contain 'Date' and 'Country' columns.")
    return df


def run_eda(df: pd.DataFrame, output_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Run EDA based on the role 2 code.

    The original file mostly printed results and displayed figures.
    This integrated version saves tables and plots so they can be reused in the report and PPT.
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

    # Histogram visualization from the EDA file.
    df.hist(figsize=(16, 11))
    plt.tight_layout()
    plt.savefig(out / "numeric_histograms.png", dpi=300)
    plt.close()

    # Exclude the obvious 999 dirty value before drawing the fuel-change boxplot.
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

    # Correlation heatmap from the EDA file.
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=9)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png", dpi=300)
    plt.close()

    # Country-level local fuel price distribution. Log scale is kept because local currencies differ greatly.
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
    Replace obvious dirty values based on Jo_Minseo_Outlierreplace.py.

    The project treats 999.0 and -99.0 as input/system errors, not meaningful crisis extremes.
    General IQR outliers are not removed automatically because crisis-related extremes may be informative.
    """
    df = df.copy()

    if "Fuel_Price_Change_Percent" in df.columns:
        df["Fuel_Price_Change_Percent"] = df["Fuel_Price_Change_Percent"].replace(999.0, np.nan)

    if "News_Sentiment_Score" in df.columns:
        df["News_Sentiment_Score"] = df["News_Sentiment_Score"].replace(-99.0, np.nan)

    return df


def fill_time_series_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values based on Jo_Minseo_fillna.py.

    The submitted code used country-level forward fill and backward fill.
    To reduce future-information leakage in the final predictive pipeline,
    this version uses forward fill first, then country median and global median fallbacks.
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
    Create next-day targets that match the proposal.

    Classification target:
        Next_Day_Fuel_Rise = 1 if next day's Fuel_Price_Change_Percent is positive, else 0

    Regression target:
        Next_Day_Fuel_Change_Amount = next day's Fuel_Price_Local - today's Fuel_Price_Local
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
    Build X, classification y, and regression y.

    The Date_Ordinal idea from Kim_Dana_Standarization_Data.py is preserved,
    while future target columns are excluded from features.
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
    Combine imputation, encoding, and scaling in one sklearn preprocessor.

    Because this preprocessor is inside a Pipeline, it is fitted only on training data
    and then applied to test data, reducing data leakage.
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
    Split train and test data by time order.

    The submitted classification code used a stratified random split.
    For the final next-day prediction task, a chronological split is safer because future rows do not enter training.
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
    """Run Decision Tree classification using the role 4 model structure, revised for next-day binary prediction."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = time_based_train_test_split(X, y, test_size=test_size)
    X_train = X_train.drop(columns=["Date"], errors="ignore")
    X_test = X_test.drop(columns=["Date"], errors="ignore")
    categorical_cols = [c for c in categorical_cols if c != "Date"]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scaler_name="standard", imputer_strategy="median")
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=random_state)),
        ]
    )

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
    """Extract Decision Tree feature importance with post-preprocessing feature names."""
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
    """Calculate MAE, MSE, RMSE, and R2 for regression evaluation."""
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
    """Run multiple linear regression, revised to predict next-day fuel price change amount."""
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
    """Extract linear regression coefficients with post-preprocessing feature names."""
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
    Run K-Means as a supporting analysis based on the role 4 clustering file.

    The submitted file hardcoded best_k=8.
    This integrated version selects best_k using silhouette score.
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
    """Save the key final results as a JSON summary."""
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
    output_dir: str | Path = "term_project_integrated_outputs_en",
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
    run_cluster: bool = True,
) -> Dict[str, Any]:
    """
    Final top-level integrated function.

    It follows the end-to-end project flow:
    EDA -> preprocessing -> classification -> regression -> optional clustering -> output saving.
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
    parser.add_argument("--output-dir", default="term_project_integrated_outputs_en", help="Directory to save outputs")
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
