# JCDSI-0101_Alpha
from pathlib import Path

# Bank Marketing Campaign Subscription Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://github.com/Faharfaza/Final-Project/blob/6c25f746eb394e3adcb5b6b58de5616f562245c5/app.py )
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Library](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

**Final Project - Data Science & Machine Learning Program**  
**Institution:** Purwadhika Digital School  
**Group Members:**  
- [Kenny Wikantiyoso]  
- [Faharuddien Aunurriza]  

---

## Project Overview

Marketing campaigns for financial products often suffer from low conversion rates due to untargeted outreach. Contacting customers indiscriminately leads to inefficient resource allocation, unnecessary operational costs, and reduced campaign effectiveness.  

This project develops a machine learning classification model capable of predicting whether a client will subscribe to a term deposit before a marketing call is made. By leveraging historical campaign data, demographic attributes, and macroeconomic indicators, the model enables data-driven targeting strategies that improve conversion probability and optimize marketing efficiency.

The project follows a complete end-to-end data science pipeline, including data preprocessing, exploratory analysis, feature engineering, imbalance handling, model benchmarking, hyperparameter tuning, and interpretability analysis.

---

## Business Context and Objective

### Problem Statement

Marketing teams currently rely on broad contact strategies rather than predictive targeting. Without a forecasting mechanism, campaigns waste time contacting low-probability clients while potentially missing high-probability prospects. This reactive approach reduces return on marketing investment and increases operational workload.

---

### Objectives

1. Predictive Modeling  
   Build a binary classification model that predicts whether a client will subscribe to a term deposit.

2. Metric Optimization  
   Optimize evaluation metrics suitable for imbalanced classification, focusing on F2-Score, ROC-AUC and Precision-Recall performance.

3. Decision Support  
   Provide interpretable insights into key factors influencing customer subscription behavior.

---

## Analytical Approach

To ensure model reliability and business usability, the following methodology was implemented:

1. Data Quality and Leakage Prevention
   - Removed the feature duration because it is only known after a call ends and would cause data leakage.
   - Cleaned categorical values and standardized unknown entries instead of deleting them.

2. Preprocessing Pipeline
   - Applied stratified train-test split to preserve class distribution.
   - Performed encoding for categorical variables.
   - Implemented scaling where required by algorithm assumptions.
   - Ensured preprocessing steps were contained inside pipelines to avoid training-test contamination.

3. Handling Class Imbalance
   - The dataset contained significantly more “No” responses than “Yes.”
   - Evaluation metrics and modeling strategy prioritized minority-class detection rather than raw accuracy.

4. Model Benchmarking
   Multiple algorithms were trained and compared:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting / XGBoost

5. Hyperparameter Optimization
   - RandomizedSearchCV was used to efficiently explore parameter combinations.
   - Cross-validation ensured performance stability across folds.

---

## Key Results

### Final Model Performance

The selected final model demonstrated the strongest balance across classification metrics and the most stable cross-validation performance.

Key observations:

- Ensemble methods significantly outperformed single-estimator models.
- The chosen model achieved superior ROC-AUC and Precision-Recall scores.
- The model maintained strong minority-class detection ability without overfitting.

Interpretation:  
The final model reliably distinguishes potential subscribers from non-subscribers using only information available before contact, making it suitable for real-world deployment.

---

## Model Insights and Interpretability

Feature importance and SHAP-based interpretability analysis revealed consistent drivers across models:

- Prior campaign interactions strongly influence subscription likelihood.
- Macroeconomic indicators reflect market confidence and affect deposit decisions.
- Occupational categories and education levels show meaningful segmentation patterns.
- Behavioral history provides stronger predictive power than static demographics.

Directional SHAP analysis showed that certain job categories and client profiles increase predicted probability, while others reduce it. This confirms the model captures real behavioral heterogeneity rather than relying on a single dominant feature.

---

## Conclusion and Recommendation

### Technical Conclusion

The modeling phase demonstrates that customer subscription behavior can be predicted with reliable performance using structured client and campaign data. Ensemble algorithms provided the most stable results, confirming that combining multiple decision learners improves generalization and reduces prediction variance.

The alignment between EDA findings and model feature importance further validates that the model is learning meaningful patterns rather than noise. The final model therefore satisfies both predictive performance and interpretability requirements.

---

### Business Impact

Implementing this predictive system allows marketing teams to shift from mass outreach to targeted engagement.

Expected benefits:

- Higher campaign conversion rates
- Reduced marketing costs
- More efficient call allocation
- Improved customer experience through relevant offers

---

### Strategic Recommendations

1. Target High-Probability Segments  
   Focus outreach on customer profiles with strong predicted subscription probability.

2. Prioritize Previously Contacted Clients  
   Clients with prior campaign interaction history show higher conversion likelihood.

3. Time Campaigns with Economic Conditions  
   Macroeconomic indicators significantly influence deposit decisions, so campaign timing should align with favorable financial climates.

4. Deploy Predictive Scoring System  
   Integrate the model into CRM or campaign tools to score leads before contact.

---

## Model Deployment

The Streamlit application is fully functional in a local environment. To run the application, please execute the following command in the terminal from the project directory:

```
streamlit run app.py
```

This will launch the interactive web interface in your browser, allowing real-time predictions using the trained machine learning model.

Cloud deployment is currently in progress. In the meantime, running the application locally ensures full access to all features and model functionalities.

---

**Access the Application:**  
[https://github.com/Faharfaza/Final-Project/blob/6c25f746eb394e3adcb5b6b58de5616f562245c5/app.py]

**Access the Tableu:**
[https://public.tableau.com/shared/QR9NJGY2S?:display_count=n&:origin=viz_share_link]

---

## Repository Structure
