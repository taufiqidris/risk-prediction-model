# ICU Patient Outcome Prediction & Risk Stratification

## Overview
This repository contains an end-to-end machine learning pipeline designed to predict in-hospital mortality for ICU patients. By leveraging advanced tree-based algorithms (XGBoost) and Explainable AI (SHAP), this project provides transparent, data-driven risk stratification to support clinical decision-making. 

The analysis goes beyond standard predictive modeling by grounding the data engineering and feature selection in epidemiological and physiological realities, focusing heavily on markers of multi-organ failure and metabolic stress.

## Key Highlights
* **Domain-Aware EDA:** Treated missing laboratory values (e.g., troponin, lactate) not as random errors, but as vital indicators of clinical suspicion by engineering specific `_is_missing` features.
* **Temporal Feature Engineering:** Captured patient deterioration dynamics by calculating 'delta' features for vital signs (e.g., fluctuations in Mean Arterial Pressure and Heart Rate).
* **Imbalanced Classification Handling:** Addressed the highly skewed 14% mortality rate by utilizing `scale_pos_weight` within the XGBoost framework, explicitly optimizing for the Area Under the Precision-Recall Curve (PR-AUC) rather than misleading raw accuracy.
* **Native Model Explainability:** Bypassed standard wrapper limitations by extracting SHAP (SHapley Additive exPlanations) values directly from the XGBoost C++ backend, unpacking the "black box" to prove the model relies on sound clinical logic.

## Repository Structure
* `01_eda.ipynb`: Exploratory Data Analysis covering data cleaning, missingness evaluation, outlier detection, temporal feature engineering, and bivariate statistical analysis.
* `02_predictive_modeling.ipynb`: The machine learning pipeline featuring a scikit-learn `ColumnTransformer`, XGBoost training, rigorous PR-AUC evaluation, and SHAP visualizations.
* `icu_data_engineered.csv`: The cleaned and engineered dataset ready for model consumption. *(Note: Ensure this is added to your .gitignore if the file is too large or contains sensitive data).*

## Methodology & Results
The final XGBoost model successfully identifies high-risk patients. More importantly, the SHAP summary analysis validates that the model's predictions are primarily driven by critical indicators of systemic dysfunction, including:
* **Renal Impairment:** Elevated BUN and decreased urine output.
* **Metabolic Stress:** High lactate levels acting as a proxy for tissue hypoperfusion.
* **Hemodynamic Instability:** Severe fluctuations in arterial pressure and heart rate.

## Technologies Used
* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Seaborn, Matplotlib