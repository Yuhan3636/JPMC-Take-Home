import os
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_column_names(columns_path: str) -> List[str]:
    """
    Load column names from census-bureau.columns (one name per line).
    """
    with open(columns_path, "r") as f:
        cols = [line.strip() for line in f if line.strip()]
    # Sanity check
    if len(cols) < 2:
        raise ValueError("Column header file seems too short.")
    return cols


def load_raw_data(
    data_path: str,
    columns_path: str,
) -> pd.DataFrame:
    """
    Load the raw census data with column names.
    """
    cols = load_column_names(columns_path)
    df = pd.read_csv(
        data_path,
        header=None,
        names=cols,
        na_values=["?", " ?"],
    )
    return df


def prepare_supervised_data(
    df: pd.DataFrame,
    weight_col: str = "weight",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare X, y, weights for classification.

    - Drops rows with missing label.
    - Converts label to binary (1 = >50k; 0 = <=50k).
    - Returns:
        X: features (no label)
        y: binary labels
        w: sample weights
    """
    # Clean label
    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    # Show unique labels once so you can verify them when you run the script
    print("Unique label values:", df[label_col].unique())

    # Common encodings for this dataset:
    #  - "50000+."  (or ">50K", ">50K.")
    #  - "- 50000." (or "<=50K", "<=50K.")
    high_income_tokens = {"50000+.", ">50K", ">50K."}

    y = df[label_col].apply(lambda v: 1 if v in high_income_tokens else 0)

    # Handle weights (if missing, default to 1.0)
    if weight_col in df.columns:
        w = df[weight_col].astype(float).fillna(1.0)
    else:
        w = pd.Series(1.0, index=df.index, name="weight")

    # Features: drop label, but keep weight only as a *feature* if you want.
    # Here we exclude it as a feature and use it only as sample_weight.
    X = df.drop(columns=[label_col])

    return X, y, w


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that:
    - imputes and scales numeric features
    - imputes and one-hot encodes categorical features
    Returns (preprocessor, numeric_features, categorical_features).
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric features ({len(numeric_features)}):", numeric_features[:10], "...")
    print(f"Categorical features ({len(categorical_features)}):", categorical_features[:10], "...")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features
