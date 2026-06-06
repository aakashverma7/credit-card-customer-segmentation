# Credit Card Customer Segmentation

This project segments credit card customers based on spending and payment behaviour using unsupervised learning.

Data not included in this repository.

## Workflow

- missing-value handling
- feature scaling and log transformation
- PCA for dimensionality reduction
- KMeans model selection with silhouette analysis
- DBSCAN search for density-based alternatives

## Project Structure

- `src/customer_segmentation.py`: clustering pipeline and evaluation
- `data/README.md`: expected dataset file and schema
- `requirements.txt`: Python dependencies

## Quick Start

```bash
pip install -r requirements.txt
python src/customer_segmentation.py --input-path data/credit_card_customers.xlsx
```

## Outputs

The script writes results under `outputs/`:

- `kmeans_metrics.csv`
- `cluster_assignments.csv`
- `dbscan_metrics.csv`
