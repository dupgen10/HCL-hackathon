# 🎯 Customer Churn Prediction - HCL Hackathon

> **Sprint 0: Planning & Setup Phase** - Ideation, strategy, and roadmap for the hackathon project.

## 📋 Problem Statement

Customer Churn Prediction aims to identify customers who are likely to discontinue using a company's services. By predicting churn, businesses can take proactive measures to retain customers and reduce revenue loss.

**Requirements:**
- Build a Machine Learning prediction model to predict customer churn
- Implement techniques to handle imbalanced dataset
- Utilize appropriate evaluation metrics beyond accuracy
- Provide visualizations (Confusion Matrix / ROC Curve)

## 📊 Dataset Selection

**IBM Telco Customer Churn Dataset**

We will be using the IBM Telco Customer Churn dataset for this project, which provides a robust foundation for churn prediction modeling.

**Dataset Characteristics:**
- **Size:** 7,043 customer records with 21 features
- **Target:** Churn (Yes/No) - binary classification
- **Class Distribution:** ~26.5% churners vs ~73.5% non-churners (moderately imbalanced)
- **Source:** Available on Kaggle as `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Feature Categories:**
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Account Information:** Tenure, Contract type (Month-to-month/One-year/Two-year), PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges
- **Services Subscribed:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

**Why This Dataset:**
- Real-world telco scenario with authentic class imbalance
- Rich feature set including demographics, services, and billing data
- Well-documented and widely used for benchmarking ML approaches
- Imbalance rate (~26%)

## 🎯 Sprint 0 Activities

### ✅ Understanding the Problem Statement
- Analyzed business problem: Customer churn prediction for proactive retention
- Identified key challenge: Imbalanced dataset requiring specialized handling techniques
- Understood evaluation requirements: Beyond accuracy metrics needed (Precision, Recall, F1, ROC-AUC)
- Reviewed dataset structure: Binary classification task with 21 predictive features

### ✅ Brainstorming & Finalizing Solution Approach
- Proposed workflow: Data Preprocessing → Model Training → Evaluation → API Deployment
- Identified imbalance handling strategies:
  - Class weighting (`class_weight='balanced'`)
  - SMOTE/ADASYN oversampling techniques
  - RandomUnderSampling if needed
- Selected evaluation metrics: Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix
- Decided on API-first approach with FastAPI and optional React dashboard

### 📝 Listing Tasks (Backend, Frontend, etc.)

**Sprint 1 Tasks - Core Development (3-4 hours):**

**Data Preprocessing & ML Development:**
- Download IBM Telco Customer Churn dataset from Kaggle
- Clean `TotalCharges` column (handle whitespaces, convert to numeric)
- Encode categorical variables (one-hot encoding for tree models)
- Map Churn Yes/No to 1/0 for modeling
- Apply stratified train-test split to maintain class distribution
- Implement SMOTE/class weighting for imbalance handling
- Train baseline and advanced models (Logistic Regression, Random Forest, XGBoost)
- Generate evaluation metrics: Confusion Matrix, ROC Curve, PR Curve
- Analyze feature importance (identify key churn drivers)
- Save trained model for API integration

**Backend API Development:**
- Set up FastAPI application structure
- Create `/predict` endpoint for single customer churn prediction
- Implement input validation using Pydantic models (21 features)
- Load trained model and integrate with API
- Generate automatic Swagger documentation at `/docs`
- Test endpoints using Postman/Swagger UI

**Optional Frontend (if time permits):**
- Create React application with customer input form
- Integrate with backend API for real-time predictions
- Display prediction results and churn probability
- Visualize feature importance and model insights

**Sprint 2 Tasks - Refinement & Submission (1 hour):**
- Debug and refine API functionality
- Add error handling for missing/invalid inputs
- Test with various customer profiles
- Update README with final documentation and usage examples
  Deploy to cloud platform
- Prepare demo and presentation pitch

## 🛠️ Proposed Tech Stack

**Backend:**
- Python 3.8+ with FastAPI
- scikit-learn, imbalanced-learn (SMOTE)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualizations)
- XGBoost/LightGBM (advanced models)

**Frontend :**
- React.js with Material-UI or Streamlit

**Deployment :**
- Railway/Render (backend hosting)
- Vercel (frontend hosting)
- Docker (containerization)

**Tools:**
- Git & GitHub for version control
- Jupyter Notebook for exploratory analysis
- Postman/Swagger for API testing
- VS Code for development


**Directory Breakdown:**

- **`data/`** - Dataset storage
  - `raw/`: Original IBM Telco dataset
  - `processed/`: Cleaned and preprocessed data

- **`notebooks/`** - Jupyter notebooks for exploration and experimentation
  - EDA, model training, and evaluation notebooks

- **`models/`** - Saved trained models and preprocessing artifacts
  - Serialized model, scaler, and encoder files

- **`src/`** - Main source code
  - `data/`: Data loading and preprocessing modules
  - `models/`: Model training and evaluation scripts
  - `api/`: FastAPI application with routers and schemas

- **`tests/`** - Unit and integration tests
  - Test files for preprocessing, model, and API

- **`frontend/`** - Optional React dashboard
  - Standard React project structure

- **`docs/`** - Documentation and visualizations
  - Generated plots and API documentation

- **Root files** - Configuration and documentation
  - `.env.example`: Environment variable template
  - `requirements.txt`: Python dependencies
  - `Dockerfile`: Container configuration

## 🎯 Success Criteria

**Minimum Requirements:**
- [x] Problem statement understood
- [x] Dataset selected (IBM Telco Customer Churn)
- [x] Solution approach finalized
- [x] Tech stack selected
- [x] Tasks listed for all sprints
- [x] Roadmap documented in README
- [ ] ML model handling class imbalance with appropriate metrics
- [ ] Functional REST API with `/predict` endpoint
- [ ] Confusion Matrix and ROC Curve visualizations

**Stretch Goals:**
- [ ] React frontend dashboard
- [ ] Cloud deployment
- [ ] Docker containerization
- [ ] Feature importance visualization in API

## 📚 Resources

**Dataset:**
- [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Documentation:**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/)

##  Project Information

- **Team Name:** Team Nova


---

Made for HCL Python Full-Stack Hackathon 2025 🚀
