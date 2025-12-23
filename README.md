# Diamonds Price Prediction - ML Pipeline

This project implements an end-to-end Machine Learning pipeline for predicting diamond prices. The workflow includes exploratory analysis, data preprocessing, feature selection, and a comparison between different regression models.


## Dataset
The "Diamonds" dataset contains **53,940 rows** and **11 columns**.
* **Target:** `price`
* **Features:**
    * Numeric: `carat`, `depth`, `table`, `x`, `y`, `z`
    * Categorical: `cut`, `color`, `clarity`

## Elements Used in the Pipeline

### 1. Preprocessing and Data Cleaning
Before training the models, the following transformers were applied:
* **Data Cleaning:** Removal of rows with null dimensions (x, y, z = 0).
* **Encoding (Categorical Variables):** **Manual Mapping** was used to preserve the hierarchical order of ordinal variables (e.g., *Fair < Good < Ideal*).
* **Scaling:** Use of **StandardScaler** to normalize numeric values.

### 2. Feature Selection
* **LASSO Regression:** Used not as a final model, but as a preliminary analysis tool (with Cross Validation) to identify low-impact features.
    * *Outcome:* Based on the analysis, it was decided to retain all original features.

### 3. Regression Models (Regressors)
Three main approaches were trained and compared:

* **Ridge Regression (Linear Baseline):**
    * Linear model with L2 regularization.
    * Used as a simple starting point.
    * Limitation: Struggles to capture non-linear relationships (e.g., the exponential growth of price/carat).

* **Random Forest Regressor (Main Model):**
    * Decision tree-based model.
    * **Optimization:** Hyperparameters optimized via **Randomized Search CV** (10 folds).
    * Target parameters: `n_estimators` (number of trees) and `max_depth` (maximum depth to prevent overfitting).

* **Voting Regressor (Ensemble):**
    * A meta-estimator combining predictions from **Ridge** and **Random Forest**.
    * Goal: To attempt to average out the errors of individual models (heterogeneous approach).

### 4. Explainability (XAI)
To interpret the predictions of the "Black Box" model (Random Forest), **SHAP** (SHapley Additive exPlanations) was used:
* **Summary Plot:** To visualize global feature importance.
* **Waterfall Plot:** To explain individual predictions.
* **Dependence Plot:** To analyze the relationship between a feature's value and the target.

## Results (R² Metrics)

The final comparison on test data produced the following accuracy scores (R²):

| Model | R² Score | Notes |
| :--- | :--- | :--- |
| **Random Forest** | **0.9835** | Best performance |
| Voting Regressor | 0.9663 | Penalized by the linear component |
| Ridge Regression | 0.9125 | Baseline |

The complete flow was finally integrated into a **Scikit-Learn Pipeline** (`Pipeline` class) which includes `ColumnTransformer` for preprocessing and the final Random Forest regressor.
