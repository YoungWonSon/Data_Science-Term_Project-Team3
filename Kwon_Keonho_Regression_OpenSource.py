"""
Regression Modeling & Open Source SW Lead

Purpose:
- Predict the concrete next-day local fuel price change amount.
- Integrate preprocessing, regression modeling, model comparison,
  evaluation, and output saving into one reusable top-level function.

Target Variable:
- Next_Day_Fuel_Change_Amount
  = next day's Fuel_Price_Local - today's Fuel_Price_Local

Main Model:
- Multiple Linear Regression

Optional Model:
- Polynomial Regression Degree 2

Evaluation Metrics:
- MAE, MSE, RMSE, R2

Referenced and Extended from Internal Team reference.


Required Dataset:
    Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv
"""

import os
import warnings
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler


warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Internal team references are also stored as a dictionary so that the
# code itself clearly documents which team modules influenced each part.
# ---------------------------------------------------------------------
TEAM_CODE_REFERENCES = {
    "EDA_and_dirty_value_detection": "Seo_Jangwon_EDA.py",
    "missing_value_handling": "Jo_Minseo_fillna.py",
    "one_hot_encoding": "Jo_Minseo_Onehotencoding.py",
    "scaling_strategy": "Jo_Minseo_Scaling.py, Kim_Dana_Standarization_Data.py",
    "outlier_handling": "Jo_Minseo_Outlier.py, Jo_Minseo_Outlierdelete.py",
    "baseline_regression": "KIm_Dana_Multi-Linear-Regression.py",
}


# ---------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------

def make_one_hot_encoder() -> OneHotEncoder:
    """
    OneHotEncoder that works across different scikit-learn versions.

    Internal reference : Jo_Minseo_Onehotencoding.py.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def clean_dirty_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean obvious error-type dirty values.

    Internal references:
    - Seo_Jangwon_EDA.py: checked abnormal values such as 999.0 and -99.0.
    - Jo_Minseo_Outlier.py: used IQR-based outlier detection logic.
    - Jo_Minseo_Outlierdelete.py: handled 999.0 and -99.0 as abnormal values.

    In this final regression pipeline, the representative dirty values are
    converted to NaN first, then filled later by the preprocessing process.
    Extreme values are clipped to reduce influence on regression.
    """
    df = df.copy()

    if "Fuel_Price_Change_Percent" in df.columns:
        df.loc[df["Fuel_Price_Change_Percent"] == 999.0, "Fuel_Price_Change_Percent"] = np.nan
        df["Fuel_Price_Change_Percent"] = df["Fuel_Price_Change_Percent"].clip(lower=-20, upper=20)

    if "News_Sentiment_Score" in df.columns:
        df.loc[df["News_Sentiment_Score"] == -99.0, "News_Sentiment_Score"] = np.nan
        df["News_Sentiment_Score"] = df["News_Sentiment_Score"].clip(lower=-1, upper=1)

    return df



def time_series_fill_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric time-series values by country.

    Internal reference:Jo_Minseo_fillna.py.

    Extension:
    - Linear interpolation is applied first to reflect the time-series trend.
    - Forward-fill and backward-fill are applied after interpolation.
    - Global median replacement is used only if missing values still remain.
    """
    df = df.copy()

    if "Date" not in df.columns or "Country" not in df.columns:
        raise ValueError("The dataset must contain 'Date' and 'Country' columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = df.groupby("Country")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df[col] = df.groupby("Country")[col].transform(lambda s: s.ffill().bfill())

        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df



def add_regression_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target:
        Next_Day_Fuel_Change_Amount
        = next day's Fuel_Price_Local - today's Fuel_Price_Local
    """
    df = df.copy()

    required_cols = {"Country", "Date", "Fuel_Price_Local"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns for target generation: {missing_cols}")

    df = df.sort_values(["Country", "Date"]).reset_index(drop=True)
    df["Next_Day_Fuel_Price_Local"] = df.groupby("Country")["Fuel_Price_Local"].shift(-1)
    df["Next_Day_Fuel_Change_Amount"] = (
        df["Next_Day_Fuel_Price_Local"] - df["Fuel_Price_Local"]
    )

    # The last row of each country has no next-day value, so it cannot be used.
    df = df.dropna(subset=["Next_Day_Fuel_Change_Amount"]).reset_index(drop=True)

    return df



def build_regression_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    imputer_strategy: str = "median",
    scaler_name: str = "standard",
    polynomial_degree: int = 1,
) -> Pipeline:
    """
    Regression pipeline.

    Internal references:
    - Jo_Minseo_Scaling.py: compared RobustScaler and StandardScaler.
    - Kim_Dana_Standarization_Data.py: used StandardScaler and country dummy variables.
    - Jo_Minseo_Onehotencoding.py: used one-hot encoding for Country.
    - KIm_Dana_Multi-Linear-Regression.py: used LinearRegression as baseline model.

    Parameters
    ----------
    numeric_cols : list
        Numeric feature column names.
    categorical_cols : list
        Categorical feature column names.
    imputer_strategy : {mean, median}
        Missing value replacement strategy for numeric variables.
    scaler_name : {standard, robust}
        Scaling method for numeric variables.
    polynomial_degree : int
        1 means Multiple Linear Regression.
        2 means Polynomial Regression with degree 2.
    """
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_name must be either 'standard' or 'robust'.")

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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("polynomial", PolynomialFeatures(degree=polynomial_degree, include_bias=False)),
            ("regressor", LinearRegression()),
        ]
    )

    return model



def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Internal reference:
    - Extended from KIm_Dana_Multi-Linear-Regression.py : evaluated regression using MSE, RMSE, and R2.

    Extension:
    - MAE is added for easier interpretation of average absolute prediction error.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }



def get_feature_coefficients(fitted_model: Pipeline, top_n: int = 30) -> pd.DataFrame:
    """
    Extract regression coefficients from a fitted pipeline.

    Internal reference:
    - Extended from KIm_Dana_Multi-Linear-Regression.py : printed regression coefficients for interpretation.
    """
    preprocessor = fitted_model.named_steps["preprocess"]
    polynomial = fitted_model.named_steps["polynomial"]
    regressor = fitted_model.named_steps["regressor"]

    try:
        base_feature_names = preprocessor.get_feature_names_out()
        feature_names = polynomial.get_feature_names_out(base_feature_names)
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(regressor.coef_))]

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": regressor.coef_,
            "Abs_Coefficient": np.abs(regressor.coef_),
        }
    ).sort_values("Abs_Coefficient", ascending=False)

    return coef_df.head(top_n)



def save_actual_vs_predicted_plot(y_test: pd.Series, y_pred: np.ndarray, output_path: str) -> None:
    """
    Save an actual-vs-predicted scatter plot.

    Internal reference:
    - Extended from KIm_Dana_Multi-Linear-Regression.py : Visualized actual and predicted regression values with a diagonal reference line.
    """
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)

    min_value = min(float(np.min(y_test)), float(np.min(y_pred)))
    max_value = max(float(np.max(y_test)), float(np.max(y_pred)))
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.title("Actual vs Predicted: Next-Day Fuel Change Amount")
    plt.xlabel("Actual Change Amount")
    plt.ylabel("Predicted Change Amount")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# 2. Single top-level function for Open Source SW contribution
# ---------------------------------------------------------------------

def run_fuel_price_regression_pipeline(
    csv_path: str = "Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv",
    test_size: float = 0.2,
    cv: int = 5,
    random_state: int = 42,
    output_dir: str = "regression_outputs",
    save_outputs: bool = True,
    include_polynomial: bool = False,
) -> Dict[str, Any]:
    """
    Run the full regression modeling process under one top-level function.

    Open-source software contribution;
    Executing preprocessing, training/testing, model-combination comparison,
    evaluation, and output saving.

    Parameters
    ----------
    csv_path : str, default='Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv'
        Path to the dirty dataset CSV file.
    test_size : float, default=0.2
        Ratio of test data.
    cv : int, default=5
        Number of folds for cross-validation.
    random_state : int, default=42
        Random seed for reproducibility.
    output_dir : str, default='regression_outputs'
        Folder where result files are saved.
    save_outputs : bool, default=True
        If True, save result CSV files and plot image.
    include_polynomial : bool, default=False
        If True, also compare Polynomial Regression degree 2.

    Returns
    -------
    result : dict
        Dictionary containing : fitted best model, result tables, metrics,predictions, 
        coefficient information, and internal team references
    """
    # Step 1. Load and clean data
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = clean_dirty_values(df)
    df = time_series_fill_by_country(df)
    df = add_regression_target(df)

    target_col = "Next_Day_Fuel_Change_Amount"

    # Step 2. Candidate feature set
    base_drop_cols = [target_col, "Next_Day_Fuel_Price_Local", "Date"]

    feature_sets = {
        "all_features": [],
        "without_current_change_percent": ["Fuel_Price_Change_Percent"],
        "without_Brent_for_multicollinearity": ["Brent_Crude_USD_per_barrel"],
        "without_WTI_for_multicollinearity": ["WTI_Crude_USD_per_barrel"],
    }

    polynomial_degrees = [1, 2] if include_polynomial else [1]
    y = df[target_col]

    # Step 3. Model comparison.
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }

    kfold = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    result_rows = []
    fitted_models = []

    for feature_set_name, feature_drop_cols in feature_sets.items():
        X = df.drop(columns=base_drop_cols + feature_drop_cols, errors="ignore")

        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        for imputer_strategy in ["median", "mean"]:
            for scaler_name in ["standard", "robust"]:
                for polynomial_degree in polynomial_degrees:
                    model = build_regression_pipeline(
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                        imputer_strategy=imputer_strategy,
                        scaler_name=scaler_name,
                        polynomial_degree=polynomial_degree,
                    )

                    cv_result = cross_validate(
                        model,
                        X_train,
                        y_train,
                        cv=kfold,
                        scoring=scoring,
                        n_jobs=1,
                        return_train_score=False,
                    )

                    model.fit(X_train, y_train)
                    y_pred_test = model.predict(X_test)
                    test_metrics = evaluate_regression(y_test, y_pred_test)

                    row = {
                        "feature_set": feature_set_name,
                        "imputer_strategy": imputer_strategy,
                        "scaler_name": scaler_name,
                        "polynomial_degree": polynomial_degree,
                        "model_name": (
                            "Multiple Linear Regression"
                            if polynomial_degree == 1
                            else "Polynomial Regression Degree 2"
                        ),
                        "CV_MAE": -cv_result["test_MAE"].mean(),
                        "CV_RMSE": -cv_result["test_RMSE"].mean(),
                        "CV_R2": cv_result["test_R2"].mean(),
                        "Test_MAE": test_metrics["MAE"],
                        "Test_MSE": test_metrics["MSE"],
                        "Test_RMSE": test_metrics["RMSE"],
                        "Test_R2": test_metrics["R2"],
                    }

                    result_rows.append(row)
                    fitted_models.append(
                        {
                            "row": row,
                            "model": model,
                            "X_test": X_test,
                            "y_test": y_test,
                        }
                    )

    results_df = pd.DataFrame(result_rows).sort_values(
        by=["CV_RMSE", "Test_RMSE"],
        ascending=True,
    ).reset_index(drop=True)

    top5_results = results_df.head(5).copy()
    best_params = results_df.iloc[0].to_dict()

    best_entry = None
    for entry in fitted_models:
        row = entry["row"]
        if (
            row["feature_set"] == best_params["feature_set"]
            and row["imputer_strategy"] == best_params["imputer_strategy"]
            and row["scaler_name"] == best_params["scaler_name"]
            and row["polynomial_degree"] == best_params["polynomial_degree"]
        ):
            best_entry = entry
            break

    if best_entry is None:
        raise RuntimeError("Best model could not be found.")

    best_model = best_entry["model"]
    X_test = best_entry["X_test"]
    y_test = best_entry["y_test"]

    y_pred = best_model.predict(X_test)
    final_metrics = evaluate_regression(y_test, y_pred)

    predictions_df = pd.DataFrame(
        {
            "Actual_Next_Day_Change_Amount": y_test.values,
            "Predicted_Next_Day_Change_Amount": y_pred,
            "Residual": y_test.values - y_pred,
        },
        index=y_test.index,
    )

    coefficients_df = get_feature_coefficients(best_model, top_n=30)

    # Step 4. Save output files.
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(
            os.path.join(output_dir, "all_regression_model_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        top5_results.to_csv(
            os.path.join(output_dir, "top5_regression_model_results.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        predictions_df.to_csv(
            os.path.join(output_dir, "regression_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        coefficients_df.to_csv(
            os.path.join(output_dir, "top30_regression_coefficients.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        save_actual_vs_predicted_plot(
            y_test,
            y_pred,
            os.path.join(output_dir, "actual_vs_predicted.png"),
        )

    return {
        "best_model": best_model,
        "best_params": best_params,
        "final_metrics": final_metrics,
        "all_results": results_df,
        "top5_results": top5_results,
        "predictions": predictions_df,
        "coefficients": coefficients_df,
        "data_shape_after_preprocessing": df.shape,
        "target_column": target_col,
        "internal_team_references": TEAM_CODE_REFERENCES,
    }


# ---------------------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    result = run_fuel_price_regression_pipeline(
        csv_path="Iran_War_Global_Fuel_Crisis_Dirty_Dataset.csv",
        test_size=0.2,
        cv=5,
        random_state=42,
        output_dir="regression_outputs",
        save_outputs=True,
        include_polynomial=False,
    )

    print("\n==============================")
    print("Regression Modeling & Open Source SW Lead")
    print("==============================")

    print("\n[Internal Team Code References]")
    for key, value in result["internal_team_references"].items():
        print(f"- {key}: {value}")

    print("\n==============================")
    print("Best Regression Model")
    print("==============================")
    for key, value in result["best_params"].items():
        print(f"{key}: {value}")

    print("\n==============================")
    print("Final Test Metrics")
    print("==============================")
    for key, value in result["final_metrics"].items():
        print(f"{key}: {value:.4f}")

    print("\n==============================")
    print("Top 5 Model Combinations")
    print("==============================")
    print(result["top5_results"])

    print("\n==============================")
    print("Top 30 Regression Coefficients")
    print("==============================")
    print(result["coefficients"])

    print("\nOutput files are saved in: regression_outputs/")
