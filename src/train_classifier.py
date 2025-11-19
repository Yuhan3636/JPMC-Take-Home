import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from data_utils import load_raw_data, prepare_supervised_data, build_preprocessor


def train_and_evaluate_classifier(
    data_path: str,
    columns_path: str,
    model_out_path: str = "classifier.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 1. Load data
    print("Loading data...")
    df = load_raw_data(data_path, columns_path)
    print(f"Raw shape: {df.shape}")

    # 2. Prepare supervised data
    X, y, w = prepare_supervised_data(df)

    # 3. Build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 4. Train-test split (stratified)
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # 5. Define classifier
    clf = GradientBoostingClassifier(
        random_state=random_state,
    )

    # 6. Full pipeline: preprocessing + model
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", clf),
    ])

    # 7. Fit model using sample weights to respect survey design
    print("Training classifier...")
    model.fit(X_train, y_train, clf__sample_weight=w_train)

    # 8. Evaluate
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # 9. Save model
    print(f"\nSaving trained model to {model_out_path}")
    joblib.dump(model, model_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/census-bureau.data",
        help="Path to censusbureau.data",
    )
    parser.add_argument(
        "--columns-path",
        type=str,
        default="data/census-bureau.columns",
        help="Path to census-bureau.columns",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="classifier.joblib",
        help="Where to save the trained model",
    )
    args = parser.parse_args()

    train_and_evaluate_classifier(
        data_path=args.data_path,
        columns_path=args.columns_path,
        model_out_path=args.model_out,
    )
