# Results

Run on 2026-09-26.

## Data

- Input file: `CC GENERAL.csv` (8,950 rows x 18 columns)
- Rows after dropping duplicate `CUST_ID`: 8,950
- Missing CREDIT_LIMIT filled with its median; missing MINIMUM_PAYMENTS filled with its own median: CREDIT_LIMIT 1, MINIMUM_PAYMENTS 313 (medians used: CREDIT_LIMIT 3,000.00, MINIMUM_PAYMENTS 312.34)
- Rows with PAYMENTS = 0: 240; missing MINIMUM_PAYMENTS among them: 240
- Rows where MINIMUM_PAYMENTS > PAYMENTS (before filling): 2,365
- Ratio values clipped to [0, 1]: CASH_ADVANCE_FREQUENCY 8
- TENURE (months of card service): 6 to 12; counts: 6: 204; 7: 190; 8: 196; 9: 175; 10: 236; 11: 365; 12: 7,584
- Rows where PURCHASES = ONEOFF_PURCHASES + INSTALLMENTS_PURCHASES (within 0.01): 8,931 (99.8%)
- Rows used for clustering: 8,950
- Split: none (unsupervised). KMeans is fit on all 8,950 rows. For stability, KMeans is refit on 40 bootstrap resamples of 8,950 rows; each refit labels all 8,950 rows and the labelings are compared pairwise (780 pairs) with the adjusted Rand index (ARI).

## Features

- 17 numeric columns; log1p on 10 amount and count columns; StandardScaler on all.
- PCA keeping 90% of the variance: 8 components, 91.5% explained.
- Variance per component: 0.339, 0.220, 0.094, 0.074, 0.066, 0.050, 0.042, 0.031

## Choosing k

Silhouette is computed in PCA space (on the 8 kept components). Stability is the mean and the standard deviation of the pairwise ARIs over the 780 pairs of bootstrap refits (n_boot = 40).

| k | Inertia | Silhouette (PCA space) | Stability (mean ARI) | Stability (SD of pairwise ARIs) | Stability (worst pair) | Smallest cluster (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 100,416 | 0.274 | 0.977 | 0.012 | 0.933 | 35.4 |
| 3 | 82,920 | 0.246 | 0.958 | 0.017 | 0.897 | 32.0 |
| 4 | 73,850 | 0.239 | 0.767 | 0.204 | 0.477 | 17.3 |
| 5 | 65,523 | 0.243 | 0.937 | 0.028 | 0.840 | 14.6 |
| 6 | 59,618 | 0.242 | 0.957 | 0.011 | 0.926 | 11.3 |
| 7 | 55,873 | 0.246 | 0.880 | 0.096 | 0.700 | 5.5 |
| 8 | 52,494 | 0.240 | 0.893 | 0.100 | 0.530 | 4.6 |
| 9 | 49,584 | 0.207 | 0.892 | 0.095 | 0.646 | 4.6 |
| 10 | 46,976 | 0.208 | 0.918 | 0.047 | 0.665 | 4.5 |

Chosen k: 6

## Clusters (k = 6)

| Cluster | Customers | Share (%) | Label |
| ---: | ---: | ---: | --- |
| 1 | 2,260 | 25.3 | High cash advances, low purchase frequency, low instalment purchases |
| 2 | 1,673 | 18.7 | High instalment purchases, high purchase frequency, low cash advances |
| 3 | 1,423 | 15.9 | High one-off purchases, high purchase frequency, high instalment purchases |
| 4 | 1,388 | 15.5 | High one-off purchases, low instalment purchases, low cash advances |
| 5 | 1,197 | 13.4 | High cash advances, high balance, high purchase frequency, high payments |
| 6 | 1,009 | 11.3 | Low balance-update frequency, low balance, low payments |

Label rule: among BALANCE, BALANCE_FREQUENCY, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES, PURCHASES_FREQUENCY, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS, PRC_FULL_PAYMENT, take the (up to) 3 features whose cluster mean after log1p and scaling is furthest from the overall mean, keeping only those at least 0.5 SD away; add a fourth feature when it is also at least 0.5 SD away and its |z| is within 0.05 of the third. "high" terms are listed before "low" terms.

## k = 6 against k = 3

Clusters at k = 3 (same labelling rule):

| Cluster | Customers | Share (%) | Label |
| ---: | ---: | ---: | --- |
| 1 | 3,210 | 35.9 | High purchase frequency, high one-off purchases, high instalment purchases |
| 2 | 2,877 | 32.1 | High cash advances, low purchase frequency, low instalment purchases |
| 3 | 2,863 | 32.0 | Low balance, low cash advances, low balance-update frequency |

Customers by k = 3 cluster (rows) and k = 6 cluster (columns):

| k = 3 cluster | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 | Cluster 5 | Cluster 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 400 | 1,389 | 530 | 891 | 0 |
| 2 | 2,260 | 3 | 0 | 242 | 299 | 73 |
| 3 | 0 | 1,270 | 34 | 616 | 7 | 936 |

Cluster profiles at k = 3 (mean per cluster, original units). The MINIMUM_PAYMENTS means include the median-filled values.

| Feature | Cluster 1 | Cluster 2 | Cluster 3 | All |
| --- | ---: | ---: | ---: | ---: |
| BALANCE | 2,044 | 2,367 | 221 | 1,564 |
| BALANCE_FREQUENCY | 0.98 | 0.92 | 0.72 | 0.88 |
| PURCHASES | 2,294 | 104 | 459 | 1,003 |
| ONEOFF_PURCHASES | 1,422 | 86 | 170 | 592 |
| INSTALLMENTS_PURCHASES | 872 | 18 | 289 | 411 |
| CASH_ADVANCE | 763 | 2,157 | 37 | 979 |
| CREDIT_LIMIT | 5,760 | 4,288 | 3,282 | 4,494 |
| PAYMENTS | 2,629 | 1,765 | 696 | 1,733 |
| MINIMUM_PAYMENTS | 1,193 | 1,081 | 217 | 845 |
| PURCHASES_FREQUENCY | 0.83 | 0.07 | 0.52 | 0.49 |
| CASH_ADVANCE_FREQUENCY | 0.10 | 0.29 | 0.01 | 0.13 |
| PRC_FULL_PAYMENT | 0.16 | 0.03 | 0.27 | 0.15 |
| TENURE | 11.8 | 11.4 | 11.4 | 11.5 |
| MINIMUM_PAYMENTS imputed (%) | 0.5 | 2.3 | 8.0 | 3.5 |

## Cluster profiles (mean per cluster, original units)

Amounts are in the dataset's currency units; frequencies and PRC_FULL_PAYMENT are fractions; TENURE is in months. The MINIMUM_PAYMENTS means include the median-filled values.

| Feature | Cluster 1 | Cluster 2 | Cluster 3 | Cluster 4 | Cluster 5 | Cluster 6 | All |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BALANCE | 2,322 | 405 | 1,564 | 1,176 | 3,493 | 37 | 1,564 |
| BALANCE_FREQUENCY | 0.93 | 0.92 | 0.97 | 0.94 | 0.98 | 0.34 | 0.88 |
| PURCHASES | 24 | 661 | 3,424 | 704 | 1,434 | 250 | 1,003 |
| ONEOFF_PURCHASES | 21 | 37 | 2,243 | 640 | 810 | 142 | 592 |
| INSTALLMENTS_PURCHASES | 3 | 625 | 1,181 | 64 | 624 | 108 | 411 |
| CASH_ADVANCE | 2,103 | 44 | 57 | 161 | 2,906 | 151 | 979 |
| CREDIT_LIMIT | 4,175 | 2,879 | 6,997 | 3,738 | 6,038 | 3,569 | 4,494 |
| PAYMENTS | 1,705 | 777 | 3,145 | 1,070 | 3,104 | 675 | 1,733 |
| MINIMUM_PAYMENTS | 1,056 | 605 | 782 | 685 | 1,633 | 143 | 845 |
| PURCHASES_FREQUENCY | 0.02 | 0.84 | 0.92 | 0.36 | 0.72 | 0.26 | 0.49 |
| CASH_ADVANCE_FREQUENCY | 0.29 | 0.01 | 0.01 | 0.03 | 0.37 | 0.02 | 0.13 |
| PRC_FULL_PAYMENT | 0.03 | 0.33 | 0.31 | 0.05 | 0.04 | 0.20 | 0.15 |
| TENURE | 11.4 | 11.5 | 11.9 | 11.6 | 11.5 | 11.2 | 11.5 |
| MINIMUM_PAYMENTS imputed (%) | 1.3 | 1.1 | 0.4 | 2.8 | 0.4 | 21.4 | 3.5 |

## Setup

- Random seed: 42 (KMeans random_state and bootstrap resampling); KMeans n_init = 20
- Python 3.13.13; numpy 2.5.3; pandas 3.0.6; scikit-learn 1.9.1; matplotlib 3.11.2; openpyxl 3.1.5
- Runtime: 559 s
