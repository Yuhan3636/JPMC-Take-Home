import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

from data_utils import load_raw_data, prepare_supervised_data, build_preprocessor


def fit_kmeans_segmentation(
    data_path: str,
    columns_path: str,
    n_clusters: int = 5,
    random_state: int = 42,
):
    # 1. Load data
    print("Loading data for segmentation...")
    df = load_raw_data(data_path, columns_path)
    print(f"Raw shape: {df.shape}")

    # 2. Reuse supervised prep to get X and weights (but we ignore y here)
    X, y, w = prepare_supervised_data(df)

    # 3. Build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # 4. Build clustering pipeline
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state,
    )

    cluster_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("kmeans", kmeans),
    ])

    # 5. Fit clustering model (weighted by survey weight)
    print(f"Fitting KMeans with k={n_clusters}...")
    cluster_pipeline.fit(X, kmeans__sample_weight=w)

    # 6. Extract cluster assignments
    # (KMeans labels_ live in the inner estimator)
    kmeans_step = cluster_pipeline.named_steps["kmeans"]
    cluster_labels = kmeans_step.labels_
    df["cluster"] = cluster_labels

    # 7. Basic cluster profiling: size, income rate, average age, etc.
    print("\nCluster sizes (weighted):")
    weighted_sizes = df.groupby("cluster")["weight"].sum()
    print(weighted_sizes)

    print("\nCluster income rate (approx):")
    # y is 1 if high income
    df["high_income"] = y
    income_rate = df.groupby("cluster").apply(
        lambda g: np.average(g["high_income"], weights=g["weight"]),
        include_groups=False
    )
    print(income_rate)

    # A few example profile stats you can mention in the report
    profile_cols = [
        "age",
        "education",
        "marital stat",
        "sex",
        "race",
        "weeks worked in year",
    ]

    print("\nCluster profiles (selected features):")
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        print(f"\n=== Cluster {c} ===")
        print(f"Weighted size: {sub['weight'].sum():.1f}")
        print(f"High income rate: {np.average(sub['high_income'], weights=sub['weight']):.3f}")

        # Numeric example: avg age
        if "age" in sub.columns:
            print(f"Avg age (weighted): {np.average(sub['age'], weights=sub['weight']):.1f}")

        # Categorical examples: most common values
        for col in ["education", "marital stat", "sex", "race"]:
            if col in sub.columns:
                top = (
                    sub.groupby(col)["weight"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(3)
                )
                print(f"Top {col}:")
                print(top)


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
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for segmentation",
    )
    args = parser.parse_args()

    fit_kmeans_segmentation(
        data_path=args.data_path,
        columns_path=args.columns_path,
        n_clusters=args.n_clusters,
    )
