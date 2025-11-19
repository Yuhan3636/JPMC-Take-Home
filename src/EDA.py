import os
import pandas as pd
import matplotlib.pyplot as plt


def load_data(columns_path: str, data_path: str) -> pd.DataFrame:
    """Load the census dataset with column names."""
    with open(columns_path, "r") as f:
        cols = [line.strip() for line in f if line.strip()]

    df = pd.read_csv(
        data_path,
        header=None,
        names=cols,
        na_values=["?", " ?"],
    )

    # Add binary high-income flag for convenience
    df["high_income"] = (
        df["label"].astype(str).str.strip() == "50000+."
    ).astype(int)

    return df


def ensure_figures_dir(fig_dir: str = "figures") -> str:
    """Create figures directory if it does not exist."""
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def plot_age_distribution(df: pd.DataFrame, fig_dir: str):
    plt.figure()
    df["age"].hist(bins=30)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution")
    out_path = os.path.join(fig_dir, "fig_age_hist.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved age distribution to {out_path}")


def plot_income_distribution(df: pd.DataFrame, fig_dir: str):
    plt.figure()
    df["label"].value_counts().plot(kind="bar")
    plt.xlabel("Income Category")
    plt.ylabel("Count")
    plt.title("Income Label Distribution")
    out_path = os.path.join(fig_dir, "fig_income_dist.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved income distribution to {out_path}")


def plot_education_frequency(df: pd.DataFrame, fig_dir: str, top_n: int = 10):
    plt.figure()
    df["education"].value_counts().head(top_n).plot(kind="bar")
    plt.xlabel(f"Education Level (Top {top_n})")
    plt.ylabel("Count")
    plt.title("Top Education Levels by Frequency")
    out_path = os.path.join(fig_dir, "fig_education_bar.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved education frequency to {out_path}")


def plot_high_income_by_education(df: pd.DataFrame, fig_dir: str, top_n: int = 10):
    # Compute high-income rate by education
    edu_income = (
        df.groupby("education")
        .agg(
            count=("high_income", "size"),
            high_income_rate=("high_income", "mean"),
        )
        .sort_values("count", ascending=False)
    )

    top_edu = edu_income.head(top_n)

    plt.figure()
    top_edu["high_income_rate"].plot(kind="bar")
    plt.xlabel(f"Education Level (Top {top_n} by count)")
    plt.ylabel("High-income rate")
    plt.title("High-income Rate by Education Level")
    out_path = os.path.join(fig_dir, "fig_income_by_education.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved high-income-by-education plot to {out_path}")


def main():
    # Paths assuming you run this from the project root:
    # JPMC-Take-Home/
    columns_path = "data/census-bureau.columns"
    data_path = "data/census-bureau.data"

    print("Loading data...")
    df = load_data(columns_path, data_path)
    print(f"Data shape: {df.shape}")

    fig_dir = ensure_figures_dir("figures")

    # Generate figures
    plot_age_distribution(df, fig_dir)
    plot_income_distribution(df, fig_dir)
    plot_education_frequency(df, fig_dir, top_n=10)
    plot_high_income_by_education(df, fig_dir, top_n=10)

    print("All figures generated.")


if __name__ == "__main__":
    main()
