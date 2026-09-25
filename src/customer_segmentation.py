"""Segment credit card customers by how they use the card.

Cleans the 17 behaviour columns of the Kaggle "Credit Card Dataset for
Clustering", reduces them with PCA and clusters customers with KMeans. The
number of clusters is picked from a scan over k = 2..10 that reports inertia,
silhouette and bootstrap stability. Writes results/metrics.md and two figures.

    python src/customer_segmentation.py --data "data/CC GENERAL.csv"
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ID_COL = "CUST_ID"

# Right-skewed amounts and transaction counts, all >= 0.
LOG_COLS = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
]
# Documented as shares between 0 and 1.
RATIO_COLS = [
    "BALANCE_FREQUENCY",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT",
]
FEATURES = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]
PROFILE_COLS = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT",
    "TENURE",
]
# PURCHASES is left out of the labels: in nearly every row it is the sum of the two purchase types.
LABEL_NAMES = {
    "BALANCE": "balance",
    "BALANCE_FREQUENCY": "balance-update frequency",
    "ONEOFF_PURCHASES": "one-off purchases",
    "INSTALLMENTS_PURCHASES": "instalment purchases",
    "PURCHASES_FREQUENCY": "purchase frequency",
    "CASH_ADVANCE": "cash advances",
    "CREDIT_LIMIT": "credit limit",
    "PAYMENTS": "payments",
    "PRC_FULL_PAYMENT": "full-payment share",
}
LABEL_MIN_Z = 0.5
LABEL_MAX_TERMS = 3
LABEL_NEAR_Z = 0.05

SEED = 42
N_INIT = 20
N_BOOT = 40
K_VALUES = range(2, 11)
PCA_VARIANCE = 0.90
DEFAULT_K = 6
COMPARE_K = 3
DEFAULT_DATA = Path("data/CC GENERAL.csv")


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError(f"expected a .csv or .xlsx file, got {path.name}")
    missing = [c for c in [ID_COL, *FEATURES] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Fill, clip and check the 17 features; values stay in original units."""
    stats = {"rows_loaded": len(df), "n_columns": df.shape[1]}
    print(f"rows loaded: {len(df)}")

    df = df.drop_duplicates(subset=ID_COL).set_index(ID_COL)
    stats["rows_after_dedup"] = len(df)
    print(f"rows after dropping duplicate {ID_COL}: {len(df)}")

    features = df[FEATURES].astype(float)
    parts = features["ONEOFF_PURCHASES"] + features["INSTALLMENTS_PURCHASES"]
    stats["purchases_is_sum"] = int(((features["PURCHASES"] - parts).abs() <= 0.01).sum())

    # Kept for the profile table only; it is not a clustering feature.
    min_pay_imputed = features["MINIMUM_PAYMENTS"].isna()
    no_payments = features["PAYMENTS"] == 0
    stats["no_payments"] = int(no_payments.sum())
    stats["min_pay_missing_no_payments"] = int((min_pay_imputed & no_payments).sum())
    stats["min_pay_above_payments"] = int(
        (features["MINIMUM_PAYMENTS"] > features["PAYMENTS"]).sum()
    )
    n_missing = features.isna().sum()
    stats["imputed"] = {c: int(n) for c, n in n_missing.items() if n > 0}
    credit_median = float(features["CREDIT_LIMIT"].median())
    min_pay_median = float(features["MINIMUM_PAYMENTS"].median())
    stats["medians"] = {}
    if n_missing.get("CREDIT_LIMIT", 0):
        stats["medians"]["CREDIT_LIMIT"] = credit_median
    if n_missing.get("MINIMUM_PAYMENTS", 0):
        stats["medians"]["MINIMUM_PAYMENTS"] = min_pay_median
    features["CREDIT_LIMIT"] = features["CREDIT_LIMIT"].fillna(credit_median)
    features["MINIMUM_PAYMENTS"] = features["MINIMUM_PAYMENTS"].fillna(min_pay_median)
    leftover = {c: int(n) for c, n in features.isna().sum().items() if n}
    if leftover:
        raise ValueError(f"unexpected missing values after median fill: {_fmt_counts(leftover)}")
    print(
        f"rows after median-filling missing values: {len(features)} "
        f"(values filled: {_fmt_counts(stats['imputed'])}; "
        f"missing MINIMUM_PAYMENTS with PAYMENTS = 0: {stats['min_pay_missing_no_payments']})"
    )

    ratios = features[RATIO_COLS]
    n_clipped = ((ratios < 0) | (ratios > 1)).sum()
    stats["clipped"] = {c: int(n) for c, n in n_clipped.items()}
    features[RATIO_COLS] = ratios.clip(0, 1)
    clipped = {c: n for c, n in stats["clipped"].items() if n}
    print(
        f"rows after clipping ratio columns to [0, 1]: {len(features)} "
        f"(values clipped: {_fmt_counts(clipped)})"
    )
    print(
        "rows where PURCHASES = ONEOFF_PURCHASES + INSTALLMENTS_PURCHASES: "
        f"{stats['purchases_is_sum']}"
    )

    if (features[LOG_COLS] < 0).any().any():
        raise ValueError("negative amounts or counts found; log1p needs values >= 0")
    tenure = features["TENURE"].round().astype(int)
    stats["tenure_min"] = int(tenure.min())
    stats["tenure_max"] = int(tenure.max())
    stats["tenure_counts"] = {int(v): int(c) for v, c in tenure.value_counts().sort_index().items()}
    stats["rows_used"] = len(features)
    return features, min_pay_imputed, stats


def standardise(features: pd.DataFrame) -> pd.DataFrame:
    transformed = features.copy()
    transformed[LOG_COLS] = np.log1p(transformed[LOG_COLS])
    scaled = StandardScaler().fit_transform(transformed)
    return pd.DataFrame(scaled, index=features.index, columns=FEATURES)


def reduce(scaled: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED)
    return pca.fit_transform(scaled), pca


def fit_kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(X)


def scan_k(X: np.ndarray, k_values=K_VALUES, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Inertia, silhouette and bootstrap stability for each k.

    Stability: KMeans is refit on n_boot bootstrap resamples, each refit labels
    every row of X, and the adjusted Rand index is taken over all pairs of
    those labelings. The same resamples are reused for every k.
    """
    rng = np.random.default_rng(SEED)
    resamples = [rng.integers(0, len(X), len(X)) for _ in range(n_boot)]
    rows = []
    for k in k_values:
        model = fit_kmeans(X, k)
        sizes = np.bincount(model.labels_, minlength=k)
        labelings = [fit_kmeans(X[idx], k).predict(X) for idx in resamples]
        aris = [adjusted_rand_score(a, b) for a, b in combinations(labelings, 2)]
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, model.labels_),
                "stability_mean": float(np.mean(aris)),
                "stability_sd": float(np.std(aris)),
                "stability_min": float(np.min(aris)),
                "smallest_pct": 100 * sizes.min() / len(X),
            }
        )
    return pd.DataFrame(rows)


def number_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber clusters 1..k, largest first, so ids do not depend on KMeans ordering."""
    counts = np.bincount(labels)
    order = np.argsort(-counts, kind="stable")
    new_id = np.empty_like(order)
    new_id[order] = np.arange(1, len(order) + 1)
    return new_id[labels]


def label_cluster(z: pd.Series) -> str:
    """Name the strongest label features, with a fourth term if it ties the third within 0.05 SD.

    `z` is the cluster mean after log1p (amounts and counts) and StandardScaler.
    Terms are listed "high" before "low", and within that by distance from the mean.
    """
    z = z[list(LABEL_NAMES)]
    strong = z[z.abs() >= LABEL_MIN_Z]
    ranked = strong.loc[strong.abs().sort_values(ascending=False).index]
    if ranked.empty:
        return "Close to the overall average"
    n_keep = min(LABEL_MAX_TERMS, len(ranked))
    if len(ranked) > LABEL_MAX_TERMS:
        third_abs = abs(float(ranked.iloc[LABEL_MAX_TERMS - 1]))
        next_abs = abs(float(ranked.iloc[LABEL_MAX_TERMS]))
        if abs(next_abs - third_abs) <= LABEL_NEAR_Z:
            n_keep = LABEL_MAX_TERMS + 1
    keep = ranked.iloc[:n_keep]
    terms = [f"high {LABEL_NAMES[c]}" for c, v in keep.items() if v > 0]
    terms += [f"low {LABEL_NAMES[c]}" for c, v in keep.items() if v < 0]
    text = ", ".join(terms)
    return text[0].upper() + text[1:]


def profile(
    features: pd.DataFrame, scaled: pd.DataFrame, labels: np.ndarray, min_pay_imputed: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """z-means are cluster means after log1p (amounts and counts) and StandardScaler."""
    clusters = pd.Series(labels, index=features.index, name="cluster")
    sizes = clusters.value_counts().sort_index()
    z_means = scaled.groupby(clusters).mean()
    summary = pd.DataFrame(
        {
            "customers": sizes,
            "share_pct": 100 * sizes / len(clusters),
            "label": [label_cluster(z_means.loc[c]) for c in sizes.index],
        }
    )

    means = features[PROFILE_COLS].groupby(clusters).mean().T
    means["All"] = features[PROFILE_COLS].mean()
    imputed_pct = 100 * min_pay_imputed.groupby(clusters).mean()
    imputed_pct["All"] = 100 * min_pay_imputed.mean()
    means.loc["MINIMUM_PAYMENTS imputed (%)"] = imputed_pct
    return summary, means, z_means


def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = [header, rule, *rows]
    return "\n".join("| " + " | ".join(str(c) for c in line) + " |" for line in lines)


def scan_table(scan: pd.DataFrame) -> str:
    rows = [
        [
            int(r.k),
            f"{r.inertia:,.0f}",
            f"{r.silhouette:.3f}",
            f"{r.stability_mean:.3f}",
            f"{r.stability_sd:.3f}",
            f"{r.stability_min:.3f}",
            f"{r.smallest_pct:.1f}",
        ]
        for r in scan.itertuples()
    ]
    header = [
        "k",
        "Inertia",
        "Silhouette (PCA space)",
        "Stability (mean ARI)",
        "Stability (SD of pairwise ARIs)",
        "Stability (worst pair)",
        "Smallest cluster (%)",
    ]
    return md_table(header, rows)


def summary_table(summary: pd.DataFrame) -> str:
    rows = [
        [r.Index, f"{r.customers:,}", f"{r.share_pct:.1f}", r.label] for r in summary.itertuples()
    ]
    return md_table(["Cluster", "Customers", "Share (%)", "Label"], rows, align="rrrl")


def crosstab_table(crosstab: pd.DataFrame, row_k: int) -> str:
    header = [f"k = {row_k} cluster", *[f"Cluster {c}" for c in crosstab.columns]]
    rows = [[c, *[f"{n:,}" for n in crosstab.loc[c]]] for c in crosstab.index]
    return md_table(header, rows, align="r" * len(header))


def profile_table(means: pd.DataFrame) -> str:
    def fmt(feature: str, value: float) -> str:
        if feature in RATIO_COLS:
            return f"{value:.2f}"
        if feature == "TENURE" or feature.endswith("(%)"):
            return f"{value:.1f}"
        return f"{value:,.0f}"

    header = ["Feature", *[f"Cluster {c}" for c in means.columns[:-1]], "All"]
    rows = [[f, *[fmt(f, v) for v in means.loc[f]]] for f in means.index]
    return md_table(header, rows)


def _fmt_counts(counts: dict) -> str:
    return ", ".join(f"{c} {n:,}" for c, n in counts.items()) or "none"


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def plot_k_scan(scan: pd.DataFrame, chosen_k: int, n_boot: int, path: Path) -> None:
    fig = Figure(figsize=(10, 3.8), layout="constrained")
    ax_sil, ax_stab = fig.subplots(1, 2, sharex=True)
    ax_sil.plot(scan["k"], scan["silhouette"], marker="o")
    ax_stab.errorbar(
        scan["k"],
        scan["stability_mean"],
        yerr=scan["stability_sd"],
        marker="o",
        capsize=3,
    )
    for ax in (ax_sil, ax_stab):
        ax.axvline(chosen_k, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(list(scan["k"]))
        ax.set_xlabel("k (number of clusters)")
        ax.grid(alpha=0.3)
    ax_sil.set_ylabel("Silhouette")
    ax_sil.set_title("Silhouette in PCA space", fontsize=10)
    ax_stab.set_ylabel("Mean pairwise ARI")
    ax_stab.set_title(f"Stability: ARI between {n_boot} bootstrap refits", fontsize=10)
    fig.savefig(path, dpi=150)


def plot_profile_heatmap(z_means: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    data = z_means[FEATURES].T.to_numpy()
    k = data.shape[1]
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    fig = Figure(figsize=(3.5 + 0.9 * k, 6.5), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    mid = 0.55 * max(abs(vmin), abs(vmax))
    for (i, j), value in np.ndenumerate(data):
        text = "0.0" if abs(value) < 0.05 else f"{value:+.1f}"
        color = "white" if abs(value) > mid else "black"
        ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(k), [f"{c}\n({summary.loc[c, 'share_pct']:.0f}%)" for c in z_means.index])
    ax.set_yticks(range(len(FEATURES)), FEATURES, fontsize=8)
    ax.set_xlabel("Cluster (share of customers)")
    ax.set_title(
        "Cluster mean in SD from the overall mean\n(amounts and counts after log1p)", fontsize=10
    )
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.savefig(path, dpi=150)


def write_metrics(path: Path, run: dict) -> None:
    stats, pca, k, n_boot = run["stats"], run["pca"], run["k"], run["n_boot"]
    n = stats["rows_used"]
    n_pairs = n_boot * (n_boot - 1) // 2
    ratio = pca.explained_variance_ratio_
    clipped = {c: v for c, v in stats["clipped"].items() if v} or {"none": 0}
    medians = ", ".join(f"{c} {v:,.2f}" for c, v in stats["medians"].items()) or "none"
    tenure = "; ".join(f"{m}: {c:,}" for m, c in stats["tenure_counts"].items())
    packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "openpyxl"]

    sections = [
        f"""# Results

Run on {date.today().isoformat()}.

## Data

- Input file: `{run["data_path"].name}` ({stats["rows_loaded"]:,} rows x {stats["n_columns"]} \
columns)
- Rows after dropping duplicate `{ID_COL}`: {stats["rows_after_dedup"]:,}
- Missing CREDIT_LIMIT filled with its median; missing MINIMUM_PAYMENTS filled with its \
own median: {_fmt_counts(stats["imputed"])} (medians used: {medians})
- Rows with PAYMENTS = 0: {stats["no_payments"]:,}; missing MINIMUM_PAYMENTS among them: \
{stats["min_pay_missing_no_payments"]:,}
- Rows where MINIMUM_PAYMENTS > PAYMENTS (before filling): {stats["min_pay_above_payments"]:,}
- Ratio values clipped to [0, 1]: {_fmt_counts(clipped)}
- TENURE (months of card service): {stats["tenure_min"]} to {stats["tenure_max"]}; \
counts: {tenure}
- Rows where PURCHASES = ONEOFF_PURCHASES + INSTALLMENTS_PURCHASES (within 0.01): \
{stats["purchases_is_sum"]:,} ({100 * stats["purchases_is_sum"] / n:.1f}%)
- Rows used for clustering: {n:,}
- Split: none (unsupervised). KMeans is fit on all {n:,} rows. For stability, KMeans is \
refit on {n_boot} bootstrap resamples of {n:,} rows; each refit labels all {n:,} rows and \
the labelings are compared pairwise ({n_pairs} pairs) with the adjusted Rand index (ARI).

## Features

- {len(FEATURES)} numeric columns; log1p on {len(LOG_COLS)} amount and count columns; \
StandardScaler on all.
- PCA keeping {PCA_VARIANCE:.0%} of the variance: {pca.n_components_} components, \
{ratio.sum():.1%} explained.
- Variance per component: {", ".join(f"{r:.3f}" for r in ratio)}

## Choosing k

Silhouette is computed in PCA space (on the {pca.n_components_} kept components). \
Stability is the mean and the standard deviation of the pairwise ARIs over the \
{n_pairs} pairs of bootstrap refits (n_boot = {n_boot}).

{scan_table(run["scan"])}

Chosen k: {k}

## Clusters (k = {k})

{summary_table(run["summary"])}

Label rule: among {", ".join(LABEL_NAMES)}, take the (up to) {LABEL_MAX_TERMS} features \
whose cluster mean after log1p and scaling is furthest from the overall mean, keeping only \
those at least {LABEL_MIN_Z} SD away; add a fourth feature when it is also at least \
{LABEL_MIN_Z} SD away and its |z| is within {LABEL_NEAR_Z} of the third. "high" terms are \
listed before "low" terms.
"""
    ]
    if run.get("compare") is not None:
        compare_k, compare_summary, compare_means, crosstab = run["compare"]
        sections.append(
            f"""
## k = {k} against k = {compare_k}

Clusters at k = {compare_k} (same labelling rule):

{summary_table(compare_summary)}

Customers by k = {compare_k} cluster (rows) and k = {k} cluster (columns):

{crosstab_table(crosstab, compare_k)}

Cluster profiles at k = {compare_k} (mean per cluster, original units). The \
MINIMUM_PAYMENTS means include the median-filled values.

{profile_table(compare_means)}
"""
        )
    sections.append(
        f"""
## Cluster profiles (mean per cluster, original units)

Amounts are in the dataset's currency units; frequencies and PRC_FULL_PAYMENT are fractions; \
TENURE is in months. The MINIMUM_PAYMENTS means include the median-filled values.

{profile_table(run["means"])}

## Setup

- Random seed: {SEED} (KMeans random_state and bootstrap resampling); \
KMeans n_init = {N_INIT}
- Python {platform.python_version()}; {"; ".join(f"{p} {_version(p)}" for p in packages)}
- Runtime: {run["runtime_s"]:.0f} s
"""
    )
    path.write_text("".join(sections), encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KMeans segmentation of credit card customers.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Kaggle CC GENERAL.csv or an .xlsx export of it",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"number of clusters for the final model (default {DEFAULT_K})",
    )
    parser.add_argument(
        "--compare-k",
        type=int,
        default=COMPARE_K,
        help=f"second k to cross-tabulate against --k (default {COMPARE_K})",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help=f"bootstrap resamples for the stability score (default {N_BOOT})",
    )
    args = parser.parse_args(argv)
    for name in ("k", "compare_k"):
        if getattr(args, name) not in K_VALUES:
            parser.error(f"--{name.replace('_', '-')} must be between 2 and {K_VALUES.stop - 1}")
    if args.n_boot < 2:
        parser.error("--n-boot must be at least 2")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    start = time.perf_counter()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    features, min_pay_imputed, stats = clean(load_data(args.data))
    scaled = standardise(features)
    components, pca = reduce(scaled)
    ratio = pca.explained_variance_ratio_
    print(f"PCA: {pca.n_components_} components, {ratio.sum():.1%} of variance")
    print("variance per component: " + ", ".join(f"{r:.3f}" for r in ratio))

    scan = scan_k(components, n_boot=args.n_boot)
    labels = number_by_size(fit_kmeans(components, args.k).labels_)
    summary, means, z_means = profile(features, scaled, labels, min_pay_imputed)

    compare = None
    if args.compare_k != args.k:
        other = number_by_size(fit_kmeans(components, args.compare_k).labels_)
        other_summary, other_means, _ = profile(features, scaled, other, min_pay_imputed)
        crosstab = pd.crosstab(other, labels)
        compare = (args.compare_k, other_summary, other_means, crosstab)

    plot_k_scan(scan, args.k, args.n_boot, args.results_dir / "k_selection.png")
    plot_profile_heatmap(z_means, summary, args.results_dir / "cluster_profiles.png")
    runtime = time.perf_counter() - start
    run = {
        "data_path": args.data,
        "stats": stats,
        "pca": pca,
        "scan": scan,
        "k": args.k,
        "n_boot": args.n_boot,
        "summary": summary,
        "compare": compare,
        "means": means,
        "runtime_s": runtime,
    }
    write_metrics(args.results_dir / "metrics.md", run)

    print(f"\n{scan_table(scan)}\n\nk = {args.k}\n")
    print(f"{summary_table(summary)}\n")
    if compare is not None:
        print(f"{summary_table(compare[1])}\n\n{crosstab_table(compare[3], compare[0])}\n")
    print(profile_table(means))
    print(f"\nwrote {args.results_dir / 'metrics.md'} in {runtime:.0f} s")


if __name__ == "__main__":
    main()
