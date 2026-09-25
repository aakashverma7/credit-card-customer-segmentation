import math

import numpy as np
import pandas as pd

import customer_segmentation as cs


def make_customers(n: int = 180, seed: int = 0) -> pd.DataFrame:
    """Three synthetic behaviour types: big spenders, small spenders, cash-advance users."""
    rng = np.random.default_rng(seed)
    kind = np.arange(n) % 3
    spend = np.array([3000.0, 400.0, 50.0])[kind]
    cash = np.array([100.0, 50.0, 2500.0])[kind]
    df = pd.DataFrame({cs.ID_COL: [f"C{10001 + i}" for i in range(n)]})
    for col in cs.FEATURES:
        if col in cs.RATIO_COLS:
            df[col] = rng.uniform(0, 1, n).round(3)
        elif col == "TENURE":
            df[col] = rng.choice([6, 12], n, p=[0.1, 0.9])
        else:
            base = cash if "CASH" in col else spend
            df[col] = rng.lognormal(np.log(base), 0.5).round(2)
            if col.endswith("_TRX"):
                df[col] = (df[col] / 100).round()
    df.loc[0, "CREDIT_LIMIT"] = np.nan
    df.loc[1:5, "MINIMUM_PAYMENTS"] = np.nan
    df.loc[1:2, "PAYMENTS"] = 0.0
    df.loc[6:7, "CASH_ADVANCE_FREQUENCY"] = 1.25
    return df


def test_clean_fills_and_clips():
    features, min_pay_imputed, stats = cs.clean(make_customers())
    assert stats["imputed"] == {"CREDIT_LIMIT": 1, "MINIMUM_PAYMENTS": 5}
    assert stats["min_pay_missing_no_payments"] == 2
    assert stats["clipped"]["CASH_ADVANCE_FREQUENCY"] == 2
    assert int(min_pay_imputed.sum()) == 5
    assert not features.isna().any().any()
    assert features[cs.RATIO_COLS].to_numpy().max() <= 1


def test_label_rule():
    z = pd.Series(0.0, index=cs.FEATURES)
    z[["CASH_ADVANCE", "ONEOFF_PURCHASES", "BALANCE", "TENURE"]] = [1.2, -0.9, 0.3, 2.0]
    assert cs.label_cluster(z) == "High cash advances, low one-off purchases"
    assert cs.label_cluster(pd.Series(0.1, index=cs.FEATURES)) == "Close to the overall average"
    near = pd.Series(0.0, index=cs.FEATURES)
    near[["CASH_ADVANCE", "BALANCE", "PAYMENTS", "INSTALLMENTS_PURCHASES"]] = [
        1.170,
        0.832,
        0.555,
        0.553,
    ]
    assert cs.label_cluster(near) == (
        "High cash advances, high balance, high payments, high instalment purchases"
    )


def test_xlsx_and_csv_load_the_same(tmp_path):
    df = make_customers(30)
    df.to_csv(tmp_path / "cc.csv", index=False)
    df.to_excel(tmp_path / "cc.xlsx", index=False)
    from_csv = cs.load_data(tmp_path / "cc.csv")
    from_xlsx = cs.load_data(tmp_path / "cc.xlsx")
    pd.testing.assert_frame_equal(from_csv, from_xlsx, check_dtype=False)


def test_full_pipeline_on_synthetic_data(tmp_path):
    data = tmp_path / "CC GENERAL.csv"
    make_customers().to_csv(data, index=False)
    results = tmp_path / "results"

    argv = ["--data", str(data), "--results-dir", str(results)]
    cs.main([*argv, "--k", "3", "--compare-k", "2", "--n-boot", "3"])

    for name in ("metrics.md", "k_selection.png", "cluster_profiles.png"):
        assert (results / name).stat().st_size > 0
    text = (results / "metrics.md").read_text(encoding="utf-8")
    assert "nan" not in text.lower()
    assert "## k = 3 against k = 2" in text
    assert "Silhouette (PCA space)" in text
    assert "SD of pairwise ARIs" in text
    assert "TENURE (months of card service)" in text
    scan_rows = [
        line
        for line in text.split("## Choosing k")[1].split("##")[0].splitlines()
        if line.startswith("| ") and line[2].isdigit()
    ]
    assert len(scan_rows) == len(cs.K_VALUES)
    for line in scan_rows:
        values = [float(cell.replace(",", "")) for cell in line.strip("| ").split(" | ")]
        assert all(math.isfinite(v) for v in values)
