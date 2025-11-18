# 📡 IBM Telco Customer Churn Prediction

## 📌 Project Overview
This project focuses on predicting customer churn for a Telecommunications company using the **IBM Telco Customer Churn dataset**. By identifying at-risk customers early, the business can implement retention strategies to reduce revenue loss.

**Key Challenge:** The dataset is **moderately imbalanced (~26.5% Churn vs. 73.5% Retention)**. Our solution prioritizes metrics like **Recall, F1-Score, and PR-AUC** over simple Accuracy to ensure we don't miss identifying churners.

---

## 📊 Dataset Details
We are using the standard IBM Telco dataset containing **7,043 customer records** with **21 features**.

* **Target:** `Churn` (Yes/No mapped to 1/0).
* **Demographics:** Gender, Senior Citizen, Partner, Dependents.
* **Services:** Phone, Multiple Lines, Internet (DSL/Fiber), Security, Backup, Device Protection, Tech Support, Streaming.
* **Account Info:** Tenure, Contract (Month-to-month vs. One/Two year), Paperless Billing, Payment Method.
* **Financials:** `MonthlyCharges`, `TotalCharges`.

---

## 🛠 Solution Approach (System Design)

### 1. Data Cleaning & Preprocessing
* **Handling Numeric Issues:** The `TotalCharges` column contains whitespace errors for new customers (tenure=0). We will coerce these to numeric and handle missing values.
* **Categorical Encoding:** One-Hot Encoding for nominal variables (e.g., Payment Method) and Binary Encoding for Yes/No columns.
* **Scaling:** Standardization of numerical features (`Tenure`, `MonthlyCharges`) to aid model convergence.

### 2. Handling Class Imbalance
Since Churners are the minority class (~26%):
* **Technique 1:** `class_weight='balanced'` in Logistic Regression and Random Forest.
* **Technique 2:** **SMOTE (Synthetic Minority Over-sampling Technique)** to synthesize new examples of churners in the training set.

### 3. Model Architecture
* **Baseline:** Logistic Regression (for interpretability and baseline PR-AUC).
* **Advanced:** Random Forest / XGBoost (to capture non-linear relationships between `Contract Type` and `Tenure`).

### 4. Evaluation Metrics
* **Primary:** ROC-AUC and Precision-Recall AUC (PR-AUC).
* **Secondary:** F1-Score and Recall (Focus on minimizing False Negatives).

---

## 💻 Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **ML Libraries:** Scikit-learn, XGBoost, Imbalanced-learn
* **Visualization:** Matplotlib, Seaborn (for Churn Heatmaps)

---



