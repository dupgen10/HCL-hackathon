# 🧠 Customer Churn Prediction using Explainable Machine Learning

<div align="center">

**Predict. Prevent. Retain.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20RF%20%7C%20LR-orange.svg)]()

</div>

---

## 📌 Overview

Customer churn is one of the biggest challenges for businesses. Our solution predicts which customers are likely to leave (churn) and provides **actionable insights** to help companies retain them.

### ✨ Key Features

- 🧹 **Complete data preprocessing pipeline**
- ⚖️ **Imbalanced data handling** 
- 🤖 **Multiple ML models** (Random Forest, XGBoost, Logistic Regression)
- 📊 **Comprehensive evaluation metrics** (Recall, Precision, F1, ROC-AUC)
- 🌐 **FastAPI backend** with `/predict` endpoint
- 📈 **Explainable AI** (Feature Importance + Risk Levels)
- 🎨 **Optional dashboard UI**

---

## 🏗️ Solution Design

### 🔷 High-Level Architecture

```
Raw Data
   │
   ▼
Data Preprocessing (cleaning, encoding, scaling)
   │
   ▼
Imbalance Handling (SMOTE / Class Weights)
   │
   ▼
Model Training (RF / XGBoost / LR)
   │
   ▼
Model Evaluation (Recall, F1, AUC)
   │
   ▼
Model Export (.pkl)
   │
   ▼
FastAPI Backend (/predict)
   │
   ▼
Frontend Dashboard (optional)
```

### 🔷 Data Flow

```
customer_churn.csv
      ↓
Jupyter Notebook → ML Model → churn_model.pkl
      ↓
FastAPI → /predict → JSON Output
      ↓
UI Dashboard (Risk Levels + Visuals)
```

---

## 🧩 Features

### ✅ Core Features (MVP)

- ✔️ Data preprocessing (cleaning, encoding, scaling)
- ✔️ Imbalanced classification handling
- ✔️ Multiple ML models trained + comparison
- ✔️ Best model exported as `.pkl`
- ✔️ REST API for predictions
- ✔️ **Explainable output:**
  - `churn`: 0/1
  - `probability`: 0.0-1.0
  - `risk_level`: High/Medium/Low

### ✨ Bonus Features (Optional)

- 🎨 React dashboard
- 📊 Confusion matrix + ROC curve
- 📁 Batch prediction via CSV
- 📈 Feature importance charts

---

## 🗂️ Tech Stack

### Backend & ML
- **Python 3.8+**
- **FastAPI** - Modern web framework
- **scikit-learn** - ML models
- **imbalanced-learn** - SMOTE
- **XGBoost** - Gradient boosting
- **pandas, numpy** - Data manipulation
- **joblib** - Model serialization

### Frontend (Optional)
- **React.js** - UI framework
- **Material UI / Tailwind CSS** - Styling

### Deployment
- **Render / Railway** - Backend hosting
- **Vercel** - Frontend hosting

---

## 📊 Dataset Details

| Attribute | Details |
|-----------|---------|
| **Rows** | ~1000 |
| **Features** | Age, Gender, MonthlyUsageHours, Complaints, Transactions, SubscriptionType |
| **Target** | Churn (0 = No churn, 1 = Yes churn) |
| **Imbalance Ratio** | ~70% No churn, 30% churn |

---

## 🔧 Project Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

### 2️⃣ Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📘 Model Training (Jupyter Notebook)

The notebook covers:

1. 📊 Data exploration
2. 🧹 Preprocessing
3. ⚖️ Imbalance handling 
4. 🤖 Training ML models
5. 📈 Evaluation metrics
6. 💾 Exporting model

**To run:**

```bash
jupyter notebook
```

Navigate to `notebooks/churn_model.ipynb` and run all cells.

---

## 🚀 FastAPI Backend

### Start the API

```bash
uvicorn main:app --reload
```

### Open Swagger UI

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

### 📌 Example API Request

**Endpoint:** `POST /predict`

**Input:**

```json
{
  "age": 45,
  "gender": "Male",
  "monthly_usage_hours": 50,
  "num_transactions": 20,
  "subscription_type": "Premium",
  "complaints": 2
}
```

**Output:**

```json
{
  "churn": 1,
  "probability": 0.82,
  "risk_level": "High"
}
```

---

## 📈 Evaluation Metrics

We evaluate models using:

- ✅ **Accuracy**
- ✅ **Precision**
- ✅ **Recall** (critical for churn detection!)
- ✅ **F1-score**
- ✅ **ROC-AUC**

### 📊 Visualizations Included

- Confusion Matrix
- ROC Curve
- Feature Importance Chart

---

## 🧠 Explainable AI

We provide transparent predictions with:

1. **Feature importance chart** - Shows which factors drive churn
2. **Risk segmentation:**
   - `prob > 0.7` → **High Risk** 🔴
   - `0.4 < prob ≤ 0.7` → **Medium Risk** 🟡
   - `prob ≤ 0.4` → **Low Risk** 🟢

This helps businesses understand **why** a customer might churn and take targeted action.

---

## 🧪 Folder Structure

```
📁 churn-prediction/
│
├── 📁 notebooks/
│   └── churn_model.ipynb          # Model training notebook
│
├── 📁 api/
│   └── main.py                    # FastAPI backend
│
├── 📁 models/
│   └── churn_model.pkl            # Trained model
│
├── 📁 frontend/                   # Optional UI (React)
│
├── 📁 data/
│   └── customer_churn.csv         # Dataset
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## 🎯 MVP Highlights

✅ **Full ML lifecycle** implemented end-to-end  
✅ **Proper imbalance handling** demonstrated  
✅ **Clear and interpretable predictions**  
✅ **API + UI ready** for real-world use  
✅ **Professional architecture** + documentation  

---

## 🚀 Future Enhancements

- 🔄 **Real-time churn prediction** pipeline
- 💰 **Customer lifetime value (CLV)** estimation
- 🔁 **Auto-retraining pipeline** with new data
- 🏢 **Industry-specific churn models** (telecom, SaaS, retail)
- 🔗 **CRM integration** (Salesforce, Zendesk, Zoho)
- 📧 **Email alerts** for high-risk customers
- 📱 **Mobile app** integration

---


---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, reach out to:

- **Email:** dupgen.sherpa.ug22@nsut.ac.in , aryan.khurana.ug22@nsut.ac.in , himank.ug22@nsut.ac.in
- **GitHub:** (https://github.com/dupgen10/HCL-hackathon)
  

---

<div align="center">

**⭐ If you found this project helpful, please give it a star!**

Made with ❤️ for hackathons and learning

</div>
