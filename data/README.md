# Data

Nothing in this folder is committed except this file.

**Source:** Kaggle, "Credit Card Dataset for Clustering" (arjunbhasin2013/ccdata),
<https://www.kaggle.com/datasets/arjunbhasin2013/ccdata>. License: CC0: Public Domain.

**File:** `CC GENERAL.csv`, one row per card holder. Put it here as `data/CC GENERAL.csv`
(the script's default `--data` path). The published `results/` come from that CSV. An
`.xlsx` export of the same table also works (`--data data/CC_GENERAL.xlsx`).

## Columns used

All 18 columns are required. `CUST_ID` is the row index; the other 17 are clustering features.
Amounts are in the dataset's currency units (the source does not name the currency).

| Column | Meaning | Treatment |
|---|---|---|
| `CUST_ID` | Card holder id | Index only |
| `BALANCE` | Balance left on the account | log1p |
| `BALANCE_FREQUENCY` | How often the balance was updated (0-1) | Clipped to [0, 1] |
| `PURCHASES` | Total purchase amount | log1p |
| `ONEOFF_PURCHASES` | Purchases paid in one go | log1p |
| `INSTALLMENTS_PURCHASES` | Purchases paid in instalments | log1p |
| `CASH_ADVANCE` | Cash taken as an advance | log1p |
| `PURCHASES_FREQUENCY` | How often purchases were made (0-1) | Clipped to [0, 1] |
| `ONEOFF_PURCHASES_FREQUENCY` | How often one-off purchases were made (0-1) | Clipped to [0, 1] |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | How often instalment purchases were made (0-1) | Clipped to [0, 1] |
| `CASH_ADVANCE_FREQUENCY` | How often cash advances were taken (0-1) | Clipped to [0, 1] |
| `CASH_ADVANCE_TRX` | Number of cash-advance transactions | log1p |
| `PURCHASES_TRX` | Number of purchase transactions | log1p |
| `CREDIT_LIMIT` | Credit limit | Median fill, log1p |
| `PAYMENTS` | Amount paid by the card holder | log1p |
| `MINIMUM_PAYMENTS` | Minimum payments made | Median fill (flagged), log1p |
| `PRC_FULL_PAYMENT` | Share of full payment paid (0-1) | Clipped to [0, 1] |
| `TENURE` | Months of card service | As is |

After these steps every feature is standardised and the set is reduced with PCA. Counts of
filled and clipped values for the last run are in `results/metrics.md`.
