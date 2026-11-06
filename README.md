# Customer Segmentation Using Purchasing Behaviour

**Major project — Unsupervised Machine Learning**  
**Algorithm:** K-Means Clustering  
**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Project overview

This project identifies groups of customers with similar purchasing behaviours using an unlabeled customer dataset. The clustering features represent spending, recorded purchasing frequency, purchase recency, and product-category spending preferences. The purpose is to describe customer segments that could inform more relevant marketing decisions.

## Dataset

- **Source:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- **File:** `marketing_campaign.csv` (tab-separated)
- **Customers analyzed:** 2,240
- **Original attributes:** 29
- **Source listing license:** CC0: Public Domain

The project uses purchasing-behaviour variables and engineered features rather than a predefined target label.

## Methodology

1. **Data preparation:** Inspect missing values and duplicates; clean required purchase-related fields.
2. **Feature engineering:** Calculate total spending, recorded purchase counts, and product-category spending shares.
3. **Transformation and scaling:** Apply `log1p` to spending and purchasing counts; standardize nine selected features with `StandardScaler`.
4. **K-Means:** Compare cluster counts `k = 2` through `8`, using inertia (Elbow Method) and silhouette scores.
5. **Visualization:** Project the standardized data into two principal components using PCA for a 2D chart.
6. **Interpretation:** Compare cluster profiles and product-category expenditures.

## Results

| Metric | Result |
|---|---:|
| Customers | 2,240 |
| Selected number of clusters | 2 |
| Final silhouette score | 0.275 |
| Variance explained by first two PCA components | 53.8% |

| Cluster | Customers | Average total spending | Average recorded purchases | Average recency (days) | Largest average spending category |
|---|---:|---:|---:|---:|---|
| 0 — Higher-spending, more frequent purchasers | 1,523 (68.0%) | 834.32 | 15.60 | 49.31 | Wine |
| 1 — Lower-spending, less frequent purchasers | 717 (32.0%) | 120.39 | 6.04 | 48.69 | Meat |

**Interpretation:** The primary distinction between the groups is spending and recorded purchase count. Their average recency differs by only 0.62 days. A silhouette score of 0.275 suggests modest separation, not perfect or strongly distinct groups. PCA's 53.8% explained variance is **not** classification accuracy. Spending is reported in dataset units, not an assumed currency.

## Visual evidence

### Elbow Method
![Elbow Method](outputs/figures/elbow_method.png)

### Silhouette scores
![Silhouette Score](outputs/figures/silhouette_scores.png)

### PCA visualization
![Customer segments in PCA view](outputs/figures/pca_clusters.png)

### Average spending by cluster
![Average spending by cluster](outputs/figures/average_spending_by_cluster.png)

### Product-category spending
![Product-category spending by cluster](outputs/figures/product_preferences_by_cluster.png)

The notebooks also create exploratory data analysis charts under `outputs/figures/`.

## Project files

- `notebooks/01_data_preparation.ipynb` — Data loading, cleaning, feature engineering, EDA, and CSV export.
- `notebooks/02_clustering.ipynb` — Feature preparation, scaling, K-Means, evaluation, PCA, and clustering output.
- `notebooks/03_business_insights.ipynb` — Cluster profiles and product-category analysis.
- `data/` — Source dataset and prepared data.
- `outputs/` — Tables and saved figures.
- `requirements.txt` — Python dependencies.
- `report/` and `presentation/` — Supporting submission documents, if included.

## How to run

1. Install Anaconda (or Miniconda), then open the project folder in VS Code with the Python and Jupyter extensions.
2. Create and activate the environment:
   ```bash
   conda create -n csenv python=3.11 -y
   conda activate csenv
   python -m pip install -r requirements.txt
   python -m ipykernel install --user --name csenv --display-name "Python (csenv)"
   ```
3. Obtain `marketing_campaign.csv` from the linked Kaggle dataset and place it in `data/` if it is not included in this repository.
4. Open the notebooks in the numbered order and choose the **Python (csenv)** kernel.
5. Run Notebook 01, then 02, then 03. Check `outputs/` for resulting CSV files and PNG charts.

## Limitations

- Cluster assignments are unsupervised labels; no ground-truth customer segments or classification accuracy are available.
- The silhouette score indicates only modest separation between the two groups.
- Purchase-count features describe recorded purchases in the dataset, not a per-month frequency.
- Recommendations based on observed segments are hypotheses and have not been tested in a marketing experiment.

## Dataset attribution

Customer Personality Analysis, provided by Dr. Omar Romero-Hernandez and published on Kaggle by Akash Patel. See the dataset link above for its description and licensing information.
