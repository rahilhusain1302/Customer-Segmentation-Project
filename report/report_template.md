# Customer Segmentation Using Purchasing Behaviour

## 1. Abstract
Paste your approved proposal abstract here. After running the model, add the observed number of segments and actual findings.

## 2. Introduction and problem statement
Why businesses need to understand customer spending, product preferences, and buying frequency.

## 3. Objectives
- Identify groups of customers with similar purchase behavior.
- Analyze spending, preference shares, recency, and purchase counts.
- Suggest marketing strategies for each segment.

## 4. Dataset
- Name: Customer Personality Analysis
- Source: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
- Records and columns (record actual counts): ______
- Field meanings: six Mnt* spending fields, Recency, three channel purchase counts.
- Limitation: Purchase_Frequency is a sum of channel purchase counts, not purchases per month; Recency is measured relative to historical dataset collection, not today.

## 5. Methodology
Data import -> cleaning -> feature engineering -> log transforms -> standardization -> K-Means k=2..8 -> silhouette / elbow -> final clusters -> PCA visualization -> segment profiles.
Explain why scaling matters and why PCA is used only for plotting.

## 6. Results
- Rows after cleaning: ______
- Selected number of clusters: ______
- Silhouette score: ______
- Figure 1: outputs/figures/eda_total_spending.png
- Figure 2: outputs/figures/elbow_method.png
- Figure 3: outputs/figures/silhouette_scores.png
- Figure 4: outputs/figures/pca_clusters.png
- Figure 5: outputs/figures/average_spending_by_cluster.png
- Table: outputs/cluster_profiles.csv

## 7. Customer segments and recommendations
For each cluster, provide its number, size, typical spending, purchase count, recency, strongest product categories, descriptive label, and one action based on evidence. Do not assign labels before examining outputs.

## 8. Limitations
- Historical dataset, not live purchases.
- K-Means assumes roughly compact groups in feature space.
- Number of channels purchased is not normalized to purchases per month.
- Segment labels are researcher interpretations, not known ground truth.
- No supervised prediction, accuracy score, or causal claims.

## 9. Conclusion
Summarize observed clusters and how companies could use the findings.

## References
- Kaggle dataset source above
- Scikit-learn KMeans, StandardScaler, silhouette_score, PCA documentation.
