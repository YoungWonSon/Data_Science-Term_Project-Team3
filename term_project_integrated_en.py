import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
# English-comment version
# ============================================================
# This file integrates the code and ideas received from the team members.
#
# Reflected team roles:
# 1. Role 2 (EDA): dataset structure, missing values, outliers, statistics,
#    distributions, correlation analysis, and country-level fuel price differences
# 2. Role 3 (Preprocessing): missing-value handling, dirty-value handling,
#    Country encoding, and Robust scaling
# 3. Role 4 (Modeling): Decision Tree classification, Ridge regression,
#    and K-Means clustering
#
# PM integration decisions:
# - Keep Plan A: use Fuel_Price_Change_Percent to create a 3-class target:
#   No_Increase / Slight_Increase / Large_Increase
# - Apply the Role 4 requirement: train_test_split uses shuffle=True
# - Apply the Role 3 requirement: scaling uses RobustScaler only
# - Handle dirty values consistently:
#   Fuel_Price_Change_Percent > 100, especially 999.0, is treated as a system error
#   News_Sentiment_Score == -99.0 is also treated as a system error
# - Prevent data leakage:
#   imputation and scaling are not fitted on the full dataset.
#   They are fitted only on the training set and then applied to the test set.
#
# ------------------------------------------------------------
# Detailed integration of the original team comments and intent
# ------------------------------------------------------------
# Role 2 (Data Description and EDA) emphasized the following:
# - The purpose of EDA is to understand the dataset structure, missing values,
#   outliers, feature distributions, and relationships between variables before
#   preprocessing and modeling.
# - df.shape is used to check the number of rows and columns.
# - df.info() is used to inspect feature dtypes and non-null counts.
# - df.describe() is used to check mean, standard deviation, minimum, maximum,
#   and quartiles, which reveal scale differences and possible abnormal values.
# - isnull().sum() is used to count missing values by column.
# - Values such as 999.0 and -99.0 are treated separately because they are more
#   likely to be system-entry errors than meaningful geopolitical crisis signals.
# - Histograms show where numeric values are concentrated.
# - Boxplots show median, quartiles, and visually detectable outliers.
# - Correlation heatmaps represent relationships between variables:
#   values close to 1 indicate strong positive correlation, values close to -1
#   indicate strong negative correlation, and values close to 0 indicate weak relation.
# - Country-level Fuel_Price_Local values use very different currency units and scales,
#   so a log-scale boxplot makes small and large ranges visible together.
#
# Role 3 (Preprocessing and Feature Engineering) emphasized the following:
# - The key missing-value columns are WTI, Brent, Fuel_Price_Local,
#   Shipping_Cost_Index, and USD_Exchange_Rate.
# - Because the dataset has a country-level time-series structure, filling missing
#   values using nearby values within the same country is more meaningful than
#   using a global average.
# - The original code experimented with ffill followed by bfill. In this final
#   integration, imputation is done inside the model Pipeline using the training-set
#   median to reduce leakage from the test set.
# - Country is a nominal categorical feature, not an ordinal feature. Therefore,
#   one-hot encoding is more appropriate than label encoding.
# - IQR-based statistical outliers should not automatically be deleted because
#   large changes may be meaningful under geopolitical crisis conditions.
#   Only clear error-type values such as 999.0 and -99.0 are handled directly.
# - Scaling is necessary because models can be distorted by variables with very
#   different units and ranges.
# - This final version uses RobustScaler only, as requested. RobustScaler relies
#   on the median and IQR, so it is less sensitive to extreme values.
#
# Role 4 (Modeling and Evaluation) emphasized the following:
# - Classification groups Fuel_Price_Change_Percent into
#   No_Increase / Slight_Increase / Large_Increase.
# - Decision Trees are useful because their decision rules can be visualized and
#   explained in a presentation.
# - GridSearchCV compares parameter combinations to find the best Decision Tree setup.
# - k-fold cross validation checks whether the classification model is reliable
#   beyond one random train/test split and helps detect overfitting.
# - classification_report provides precision, recall, and F1-score, which describe
#   class-level performance beyond simple accuracy.
# - Feature importance explains which variables were most influential in the tree.
# - Regression predicts the continuous magnitude of Fuel_Price_Change_Percent.
# - MSE, RMSE, and R2 are standard metrics for regression error and explanatory power.
# - SST, SSE, and SSR are useful concepts for understanding variance decomposition
#   in regression analysis.
# - K-Means clusters similar market conditions.
# - Inertia evaluates within-cluster compactness, while silhouette score evaluates
#   both compactness and separation.
# - The elbow method and silhouette score provide evidence for selecting the number
#   of clusters.
# - PCA 2D visualization is used to show high-dimensional clustering results in a
#   presentation-friendly two-dimensional view.


DATA_FILE_NAME = "Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv"


# Multiple candidate paths are used for the dataset.
# Reason:
# - The team member files used different local paths or relative paths.
# - The final integrated script should still run even if the execution directory changes.
# - The script first checks data/ next to this file, then known project locations.
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


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs_en"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"


def make_one_hot_encoder():
    """Create a OneHotEncoder while absorbing scikit-learn version differences."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def resolve_data_path():
    """Return the first existing CSV path from the candidate list."""
    for path in DATA_PATH_CANDIDATES:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in DATA_PATH_CANDIDATES)
    raise FileNotFoundError(f"Dataset was not found. Tried:\n{tried}")


def prepare_output_dirs():
    """Create output directories for tables, metrics, and figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load the CSV file into a pandas DataFrame."""
    data_path = resolve_data_path()
    df = pd.read_csv(data_path, encoding="utf-8")
    print(f"[LOAD] Data path: {data_path}")
    print(f"[LOAD] Shape: {df.shape}")
    return df


def clean_dirty_values(df):
    """Apply consistent dirty-value handling before modeling."""
    cleaned = df.copy()

    # Date cannot be used directly as a model input string.
    # Date_Ordinal numerically represents the time flow and can be used as a feature.
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned["Date_Ordinal"] = cleaned["Date"].map(
        lambda value: value.toordinal() if pd.notna(value) else np.nan
    )

    # Sorting by country and date makes the time-series structure easier to interpret.
    cleaned = cleaned.sort_values(["Country", "Date"]).reset_index(drop=True)

    # A Fuel_Price_Change_Percent value of 999.0 is not a realistic fuel price change.
    # Since the normal maximum is around 60, values above 100 are treated as system errors.
    cleaned.loc[
        cleaned["Fuel_Price_Change_Percent"] > 100,
        "Fuel_Price_Change_Percent",
    ] = np.nan

    # A News_Sentiment_Score value of -99.0 is far outside the normal sentiment range.
    # It should not be interpreted as an extremely negative news sentiment.
    cleaned.loc[cleaned["News_Sentiment_Score"] == -99.0, "News_Sentiment_Score"] = np.nan

    return cleaned


def run_eda(raw_df, cleaned_df):
    """Run the integrated EDA workflow and save presentation-ready outputs."""
    print("[EDA] Saving descriptive tables and figures...")

    # [Role 2] Save missing values in the raw dataset.
    # This table supports the preprocessing team's decision about which columns need treatment.
    raw_df.isna().sum().rename("missing_count").to_csv(METRIC_DIR / "missing_values_raw.csv")

    # [Role 2] Save the describe() table.
    # Mean, standard deviation, minimum, and maximum reveal scale differences and abnormal values.
    raw_df.describe(include="all").to_csv(METRIC_DIR / "descriptive_statistics_raw.csv")

    # Save missing values after converting dirty values to NaN.
    # This shows how much missingness increased after handling 999.0 and -99.0.
    cleaned_df.isna().sum().rename("missing_count").to_csv(
        METRIC_DIR / "missing_values_after_dirty_value_cleaning.csv"
    )

    # [Plan A / Role 4] Save the class distribution of the 3-class target.
    # This is necessary for interpreting whether class imbalance affects model performance.
    target_counts = pd.cut(
        cleaned_df["Fuel_Price_Change_Percent"],
        bins=[-np.inf, 0, 5, np.inf],
        labels=["No_Increase", "Slight_Increase", "Large_Increase"],
        include_lowest=True,
    ).value_counts(dropna=False)
    target_counts.rename("count").to_csv(METRIC_DIR / "classification_target_counts.csv")

    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns

    # This histogram section integrates the EDA team member's numeric-distribution check.
    cleaned_df[numeric_cols].hist(figsize=(18, 12), bins=30)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_numeric_histograms.png", dpi=150)
    plt.close()

    # The correlation heatmap helps identify strongly related features,
    # such as WTI, Brent, and OPEC oil price benchmarks.
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

    # Country-level fuel prices differ heavily because local currencies differ.
    # A log-scale boxplot makes small and large price ranges visible together.
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
    """Create the 3-class classification target for Plan A."""
    result = df.copy()
    result["Fuel_Rise_Level"] = pd.cut(
        result["Fuel_Price_Change_Percent"],
        bins=[-np.inf, 0, 5, np.inf],
        labels=["No_Increase", "Slight_Increase", "Large_Increase"],
        include_lowest=True,
    )
    return result


def make_feature_matrix(df):
    """Build X by removing target columns and the raw Date column."""
    drop_cols = ["Date", "Fuel_Price_Change_Percent", "Fuel_Rise_Level"]
    return df.drop(columns=[col for col in drop_cols if col in df.columns])


def split_column_types(X):
    """Split feature columns into numeric and categorical groups."""
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X):
    """Build a ColumnTransformer for imputation, encoding, and Robust scaling."""
    numeric_cols, categorical_cols = split_column_types(X)

    # Numeric preprocessing:
    # 1. Fill missing values with the training-set median.
    #    The original Role 3 code experimented with time-series ffill/bfill,
    #    but the final evaluation should avoid leaking information from the test set.
    #    Therefore, imputation is fitted inside the Pipeline using training data only.
    # 2. Use RobustScaler only, as requested.
    #    RobustScaler uses median and IQR, making it less sensitive to extreme values.
    #    Large crisis-driven changes may be meaningful, so this approach reduces
    #    scale influence without automatically deleting all large movements.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Categorical preprocessing:
    # Country has no natural order, so one-hot encoding is appropriate.
    # Label encoding would create an artificial numeric order such as
    # USA < UK < Germany, which does not exist in the real data.
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
    """Recover feature names after preprocessing for model interpretation."""
    feature_names = list(numeric_cols)
    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())
    return feature_names


def run_classification(cleaned_df):
    """Run the integrated Decision Tree classification workflow."""
    print("[CLASSIFICATION] Training Decision Tree classifier...")

    # [Role 4] Convert Fuel_Price_Change_Percent into the Plan A target:
    # No_Increase / Slight_Increase / Large_Increase.
    # This keeps the classification direction used in the team member's code.
    model_df = add_classification_target(cleaned_df)
    model_df = model_df[model_df["Fuel_Rise_Level"].notna()].copy()

    # Fuel_Price_Change_Percent is removed from X because it was used to create y.
    # Keeping it as a feature would cause target leakage: the model would see the answer.
    X = make_feature_matrix(model_df)
    y = model_df["Fuel_Rise_Level"]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # Required integration decision for Role 4: shuffle=True.
    # stratify=y keeps the class distribution similar in train and test sets.
    # This reduces the chance that one class is over- or under-represented in the test set.
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
            # The preprocessor and model are combined in a Pipeline.
            # This ensures that imputation and scaling are fitted only on the training fold
            # during cross validation, preventing leakage inside each fold.
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )

    # Candidate Decision Tree parameters:
    # criterion: split-quality criterion
    # max_depth: maximum tree depth
    # min_samples_split: minimum samples required to split an internal node
    # min_samples_leaf: minimum samples required in a leaf node
    # class_weight: whether to compensate for class imbalance
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
    # GridSearchCV compares all parameter combinations using 5-fold CV.
    # f1_macro is used because it averages F1 across classes and rewards balanced performance.
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Store the required k-fold cross-validation result separately.
    # The selected best model is evaluated again on the training set with 5-fold accuracy.
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
    """Predict the continuous Fuel_Price_Change_Percent value."""
    print("[REGRESSION] Training Ridge regression model...")

    # The regression target is Fuel_Price_Change_Percent.
    # This reflects the corrected Role 4 regression file, where the target was changed
    # from Fuel_Price_Local to Fuel_Price_Change_Percent.
    # The 999.0 dirty values were already converted to NaN in clean_dirty_values().
    model_df = cleaned_df[cleaned_df["Fuel_Price_Change_Percent"].notna()].copy()

    X = make_feature_matrix(model_df)
    y = model_df["Fuel_Price_Change_Percent"]

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    # Keep shuffle=True for consistency with the requested Role 4 integration rule.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
    )

    regression_model = Pipeline(
        steps=[
            # As in classification, preprocessing and modeling are combined in one Pipeline
            # so the fit/transform order remains safe after the train/test split.
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)

    # MSE: average squared prediction error.
    # RMSE: square root of MSE; easier to interpret because it has the target's unit.
    # R2: proportion of target variance explained by the model.
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "model": "Ridge(alpha=1.0)",
        "target": "Fuel_Price_Change_Percent",
        "test_mse": float(mse),
        "test_rmse": rmse,
        "test_r2": float(r2),
    }
    (METRIC_DIR / "regression_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    comparison = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    comparison.to_csv(METRIC_DIR / "regression_actual_vs_predicted.csv", index=False)

    # Regression coefficients show how each feature is associated with target increase/decrease.
    # Since features are Robust-scaled, coefficients should not be interpreted in raw units.
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
    """Run K-Means as an additional pattern-discovery analysis."""
    print("[CLUSTERING] Training K-Means clustering models...")

    # [Role 4 K-Means] Use key variables that describe crisis-related market conditions.
    # Brent: international oil benchmark
    # Fuel_Price_Local: local country-level fuel price
    # Fuel_Price_Change_Percent: fuel price movement
    # Crisis_Intensity_Index: geopolitical crisis intensity
    # USD_Exchange_Rate: exchange-rate pressure
    # Inflation_Rate_Percent: macroeconomic price pressure
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

    # K-Means is distance-based, so scaling matters strongly.
    # If variables remain in different units, large-scale variables dominate distance calculation.
    # RobustScaler is used here as the only scaler, following the requested rule.
    cluster_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    X_scaled = cluster_preprocessor.fit_transform(X)

    k_results = []
    max_k = len(clustering_cols) * 2
    # Compare multiple k values.
    # Inertia measures within-cluster compactness, so lower values are better.
    # Silhouette score measures both compactness and separation; values closer to 1 are better.
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
    # The original team code fixed best_k = 8.
    # The final integrated version selects the k with the best silhouette score automatically.
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

    # PCA is used only for presentation-friendly 2D visualization.
    # K-Means itself is fitted in the original scaled feature space.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_df["Cluster"], alpha=0.7)
    plt.title(f"K-Means Clustering PCA View (k={best_k})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "clustering_pca.png", dpi=150)
    plt.close()
    
    plot_df = cluster_df.copy()
    plot_df["Date"] = pd.to_datetime(plot_df["Date"], errors="coerce")

    # Since Country is a categorical string variable,
    # convert it into numeric codes for display on the y-axis.
    countries = sorted(plot_df["Country"].unique())
    plot_df["Country_Code"] = pd.Categorical(
        plot_df["Country"],
        categories=countries,
        ordered=True
    ).codes

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        plot_df["Date"],
        plot_df["Country_Code"],
        c=plot_df["Cluster"],
        cmap="tab10" if best_k <= 10 else "tab20",
        alpha=0.75,
        s=35
    )

    plt.title(f"K-Means Clustering by Date and Country (k={best_k})")
    plt.xlabel("Date")
    plt.ylabel("Country")

    # Display country names instead of numeric country codes on the y-axis.
    plt.yticks(range(len(countries)), countries)

    # Rotate x-axis date labels to improve readability.
    plt.xticks(rotation=45)

    # Add a color bar to show the cluster number represented by each color.
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")
    cbar.set_ticks(sorted(plot_df["Cluster"].unique()))

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "clustering_date_country_scatter.png", dpi=150)
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

    return metrics


def validate_outputs():
    """Verify that the required output files were created."""
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
