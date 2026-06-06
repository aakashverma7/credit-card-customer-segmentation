import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input file must be a CSV or Excel workbook.")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "CUST_ID" in df.columns:
        df = df.set_index("CUST_ID")

    if "CREDIT_LIMIT" not in df.columns or "PAYMENTS" not in df.columns:
        raise ValueError("Dataset must include `CREDIT_LIMIT` and `PAYMENTS` columns.")

    df = df.dropna(subset=["CREDIT_LIMIT"]).copy()

    if "MINIMUM_PAYMENTS" in df.columns:
        payments_mean = df["PAYMENTS"].mean()
        minimum_payments = df["MINIMUM_PAYMENTS"].copy()
        missing_mask = minimum_payments.isna()

        zero_payment_mask = missing_mask & (df["PAYMENTS"] == 0)
        low_payment_mask = missing_mask & (df["PAYMENTS"] > 0) & (df["PAYMENTS"] < payments_mean)
        fallback_mask = missing_mask & ~(zero_payment_mask | low_payment_mask)

        minimum_payments.loc[zero_payment_mask] = 0
        minimum_payments.loc[low_payment_mask] = df.loc[low_payment_mask, "PAYMENTS"]
        minimum_payments.loc[fallback_mask] = payments_mean
        df["MINIMUM_PAYMENTS"] = minimum_payments

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.fillna(numeric_df.median())
    numeric_df = numeric_df.clip(lower=0)
    return numeric_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering analysis for credit card customers.")
    parser.add_argument("--input-path", required=True, help="Path to the customer dataset.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(load_dataset(Path(args.input_path)))
    transformed = np.log2(df + 0.01)
    scaled = StandardScaler().fit_transform(transformed)
    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(scaled)

    kmeans_rows = []
    best_k = None
    best_score = -1.0
    for n_clusters in range(2, 11):
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        labels = model.fit_predict(pca_features)
        score = silhouette_score(pca_features, labels)
        kmeans_rows.append(
            {
                "n_clusters": n_clusters,
                "inertia": model.inertia_,
                "silhouette_score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_k = n_clusters

    kmeans_metrics = pd.DataFrame(kmeans_rows).sort_values("n_clusters")
    kmeans_metrics.to_csv(output_dir / "kmeans_metrics.csv", index=False)

    best_kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    assignments = pd.DataFrame(index=df.index)
    assignments["cluster"] = best_kmeans.fit_predict(pca_features)
    assignments.to_csv(output_dir / "cluster_assignments.csv")

    dbscan_rows = []
    for eps in [0.3, 0.5, 0.7, 1.0, 1.3]:
        for min_samples in [5, 10, 15]:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(pca_features)
            non_noise_mask = labels != -1
            unique_clusters = set(labels[non_noise_mask])
            if len(unique_clusters) < 2 or non_noise_mask.sum() < 3:
                score = np.nan
            else:
                score = silhouette_score(pca_features[non_noise_mask], labels[non_noise_mask])
            dbscan_rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": len(unique_clusters),
                    "noise_ratio": float((labels == -1).mean()),
                    "silhouette_score": score,
                }
            )

    pd.DataFrame(dbscan_rows).to_csv(output_dir / "dbscan_metrics.csv", index=False)
    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
