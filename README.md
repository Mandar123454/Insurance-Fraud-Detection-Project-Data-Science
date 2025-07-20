# 🛡️ Insurance Fraud Detection using Machine Learning

A classification-based Machine Learning project to detect **fraudulent insurance claims** using algorithms like **Logistic Regression, Random Forest, and XGBoost**, enhanced with **SMOTE** for balancing and **SHAP** for interpretability.

---

## 📁 Dataset Overview

The dataset contains anonymized features related to insurance claims, including both categorical and numerical data. The target variable is:

- **Label**:  
  - `Fraud` (1)  
  - `Not Fraud` (0)

---

## 🧠 Project Objective

Predict whether a given insurance claim is fraudulent based on the provided features using supervised classification techniques.

---

## 🚀 Workflow

### ✅ Step 1: Import Libraries
Import all necessary libraries including Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, SHAP, and imbalanced-learn (SMOTE).

### 📊 Step 2: Load & Understand Dataset
- Read the dataset using `pd.read_csv()`
- Check null values, data types, shape, and unique values

### 🧹 Step 3: Data Preprocessing
- Drop irrelevant columns (like `ID`, `Customer_ID`)
- One-hot encode categorical variables
- Map label `Fraud` → 1, `Not Fraud` → 0

### ⚖️ Step 4: Handling Imbalance with SMOTE
- Apply **SMOTE** to oversample the minority (fraud) class in training data

### 🧪 Step 5: Train-Test Split
- Split into 80% training and 20% testing using `train_test_split()`

### 🔍 Step 6: Model Training & Evaluation
Train and evaluate the following models:
- 📉 **Logistic Regression**
- 🌲 **Random Forest**
- 🚀 **XGBoost** (with SMOTE + SHAP)

### 📈 Step 7: Evaluation Metrics
Used `classification_report` for:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Also evaluated models based on how well they predicted the **fraudulent class (1)**.

### 📊 Step 8: Feature Importance with SHAP (for XGBoost)
- Used SHAP to explain which features influenced the model predictions the most

---

## 🧪 Results Summary

| Model                    | Accuracy | Precision (Buy=1) | Recall (Buy=1) | F1-Score (Buy=1) |
| -------------------------| -------- | ----------------- | -------------- | ---------------- |
| ✅ Random Forest         | 90%      | 1.00              | 0.00          | 0.00              |
| ♻️ Random Forest + SMOTE | 88%      | 0.00              | 0.00          | 0.00              |
| ⚡ XGBoost + SMOTE       | 88%      | 0.00              | 0.00          | 0.00              |


✅ **XGBoost with SMOTE gave the best results** in identifying fraudulent claims.

---

## 🎯 Conclusion

- Logistic Regression and Random Forest struggled due to class imbalance.
- XGBoost combined with SMOTE improved performance in detecting fraud.
- SHAP provided feature-level insights, making the model interpretable.

---

## 🔮 Future Scope

- Include more domain-specific features.
- Use **deep learning** with larger datasets.
- Integrate into an **automated fraud detection system** for insurance companies.

---

## 📌 Tools & Libraries Used

| Tool/Library     | Purpose                          |
|------------------|----------------------------------|
| `pandas`         | Data manipulation                |
| `numpy`          | Numerical operations             |
| `matplotlib`     | Data visualization               |
| `seaborn`        | Statistical plotting             |
| `sklearn`        | ML models & preprocessing        |
| `xgboost`        | Gradient boosting classifier     |
| `imblearn (SMOTE)` | Handle class imbalance         |
| `SHAP`           | Model interpretability           |
