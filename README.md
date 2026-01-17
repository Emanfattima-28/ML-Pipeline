# End-to-End Machine Learning Pipeline for Customer Churn Prediction

## Problem Statement
Customer churn is a critical challenge for subscription-based businesses such as telecom companies. Accurately predicting which customers are likely to leave helps organizations improve retention strategies, reduce revenue loss, and enhance customer satisfaction.

The objective of this project is to design and implement an **end-to-end machine learning pipeline** using the **Scikit-learn Pipeline API** that preprocesses customer data, trains classification models, tunes hyperparameters, and predicts customer churn effectively.

---

## Dataset Description

- **Dataset Name:** Telco Churn Dataset
- **Target Variable:** `Churn`
- **Type:** Structured tabular data
- **Rows:** Varies based on dataset
- **Key Features:**
  - `tenure` – Duration of customer subscription
  - `MonthlyCharges` – Monthly billing amount
  - `TotalCharges` – Total charges incurred
  - `Contract` – Type of contract
  - `InternetService` – Internet service type
  - `PaymentMethod` – Billing method
  - `Gender`, `SeniorCitizen` – Customer demographics
  - `Churn` – Customer churn status (Yes / No)

---

## Tools & Libraries Used

- **Python**
- **NumPy & Pandas** – Data preprocessing and analysis
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn**
  - Pipeline & ColumnTransformer
  - Feature scaling and encoding
  - Model training and evaluation
  - Hyperparameter tuning
- **Joblib** – Model persistence

---

## Project Workflow & Explanation

### 1️. Data Loading & Cleaning
- Dataset loaded using Pandas.
- `customerID` column removed as it does not contribute to prediction.
- Dataset inspected using `head()`, `info()`, and `describe()`.
- Checked for missing values and duplicate records.
- Converted `TotalCharges` to numeric format.
- Removed rows containing missing values.

---

### 2️. Exploratory Data Analysis (EDA)
- Statistical summary of numerical features.
- Distribution analysis of tenure and charges.
- Analysis of churn class imbalance.
- Identification of influential features affecting churn.

---

### 3️. Feature Selection & Target Encoding
- Target variable `Churn` encoded:
  - Yes → 1
  - No → 0
- Features categorized into:
  - **Numerical Features:** `tenure`, `MonthlyCharges`, `TotalCharges`
  - **Categorical Features:** Remaining columns

---

### 4️. Feature Preprocessing using Pipeline API
- **Numerical Pipeline**
  - StandardScaler for normalization
- **Categorical Pipeline**
  - OneHotEncoder with `handle_unknown="ignore"`
- Combined using `ColumnTransformer` for efficient preprocessing.

---

### 5️. Train-Test Split
- Dataset split into:
  - **80% Training Set**
  - **20% Testing Set**
- Stratified sampling applied to preserve class balance.

---

### 6️. Model Building
Two classification models were implemented using Scikit-learn Pipelines:

#### 🔹 Logistic Regression
- Serves as a baseline model
- Effective for linear decision boundaries

#### 🔹 Random Forest Classifier
- Ensemble-based learning model
- Captures non-linear feature interactions
- More robust to noise and overfitting

---

### 7️. Hyperparameter Tuning
- **GridSearchCV** used with 5-fold cross-validation.

**Logistic Regression Parameters Tuned:**
- Regularization strength (`C`)
- Solver selection

**Random Forest Parameters Tuned:**
- Number of trees
- Maximum depth
- Minimum samples split

Best-performing models selected based on accuracy.

### 8️. Model Evaluation
Models evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Evaluation results displayed using `classification_report`.

The **Random Forest Classifier** demonstrated superior performance compared to Logistic Regression.

### 9️. Model Saving
- Best-performing pipeline saved using `joblib`.
- Entire preprocessing and model workflow stored together.
- Saved model is deployment-ready.

**Saved File:**

## Key Insights
- Customers with shorter tenure are more likely to churn.
- Higher monthly charges increase churn probability.
- Contract type and payment method strongly influence churn.
- End-to-end pipelines reduce data leakage risks.
- Random Forest performs well for complex customer behavior patterns.

## Conclusion
This project successfully demonstrates a **complete end-to-end machine learning workflow** using the **Scikit-learn Pipeline API**. By integrating preprocessing, modeling, hyperparameter tuning, and evaluation into a unified pipeline, the solution becomes scalable, reusable, and production-ready.

The **Random Forest Classifier**, combined with proper preprocessing and tuning, provides reliable and accurate churn predictions.


## Project Output
- Trained and optimized machine learning pipeline
- Saved model for future inference
- Clean and modular codebase suitable for deployment


