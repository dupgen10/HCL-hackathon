# Customer Churn Prediction System 🚀

## 📌 Project Overview
This project aims to build a Machine Learning Prediction model to identify customers likely to discontinue using a company's services. By predicting churn, businesses can take proactive measures to retain customers and reduce revenue loss.

**Key Challenge:** The dataset has a class imbalance (only ~30% churn), which this solution specifically addresses using advanced sampling techniques and specific evaluation metrics.

---

## 🧠 Problem Statement & Data
**Objective:** Predict `Churn (0 = No, 1 = Yes)` based on customer behavior and demographics.

### Dataset Features (1000 Rows):
* **Demographics:** Age (18-70), Gender (Male/Female)
* **Usage:** MonthlyUsageHours (5-200), NumTransactions (1-50)
* **Service:** SubscriptionType (Basic/Premium/Gold), Complaints (0-10)
* **Target:** Churn (Binary)

---

## 🛠 Solution Approach (System Design)

### 1. Data Preprocessing
* **Cleaning:** Handling missing values and encoding categorical variables (Gender, SubscriptionType).
* **Scaling:** Normalizing numerical features (Age, UsageHours) for model stability.

### 2. Handling Class Imbalance
Since the dataset is imbalanced (approx 70:30 ratio), we will not rely solely on Accuracy.
* **Technique:** We will use **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training data.
* **Alternative:** Using `class_weight='balanced'` in model parameters.

### 3. Model Selection
We will train and compare the following models:
* **Logistic Regression:** For baseline interpretation.
* **Random Forest / XGBoost:** To capture non-linear relationships and feature importance.

### 4. Evaluation Metrics
* **Confusion Matrix:** To visualize False Positives vs False Negatives.
* **ROC-AUC Curve:** To measure the model's ability to distinguish between classes.
* **Precision & Recall:** Critical for minimizing false negatives (missing a customer who is actually about to churn).

---

## 💻 Tech Stack
* **Language:** Python 3.x
* **ML Libraries:** Scikit-learn, Pandas, NumPy, Imbalanced-learn
* **Visualization:** Matplotlib, Seaborn
* **Deployment (Sprint 2):** Flask/FastAPI or Streamlit (for demo)

---

