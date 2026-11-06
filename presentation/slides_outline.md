# Presentation outline — one-person, eight slides

1. **Title:** Customer Segmentation Using Purchasing Behaviour, your details and project course.
2. **Problem & objectives:** Why segmenting shoppers is useful; spending, frequency and preferences.
3. **Dataset:** Kaggle Customer Personality Analysis; original vs cleaned row count, relevant columns.
4. **Pipeline:** CSV → cleaning → feature engineering → scaling → K-Means → evaluate → PCA → business insights.
5. **EDA:** 1–2 charts from `outputs/figures/eda_*.png`, with a sentence about what each means.
6. **Model selection:** Elbow and silhouette charts, chosen k, why you selected it.
7. **Results:** PCA chart and cluster profile table; discuss key differences, not arbitrary cluster IDs.
8. **Business recommendations, limitations and conclusion:** Evidence-based suggestions; dataset is historical, K-Means is not supervised classification, and clusters do not prove marketing improvements.

## Rehearsal checklist

- Explain why you didn't need labeled data.
- Explain why features had to be standardized.
- Say what k means and how you chose it.
- Explain what a silhouette score measures and why there is no accuracy score.
- Distinguish PCA plotting from K-Means training.
- Give one actual characteristic and relevant recommendation for each segment.
- Be ready to open all 3 notebooks and rerun them if a live demonstration is requested.
