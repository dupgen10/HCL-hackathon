"""
Model Training Script for Customer Churn Prediction
This script trains the model and saves all necessary files for the MVP
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Create models directory
os.makedirs('models', exist_ok=True)

print("="*60)
print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
print("="*60)

# Step 1: Load the data
print("\n[1/7] Loading data...")
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"✓ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("✗ Error: Data file not found. Please place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the current directory")
    exit(1)

# Step 2: Data Preprocessing
print("\n[2/7] Preprocessing data...")

# Drop customerID
df = df.drop('customerID', axis=1)

# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Convert SeniorCitizen to Yes/No
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

print(f"✓ Data cleaned: {df.shape[0]} rows remaining")

# Step 3: Feature Engineering
print("\n[3/7] Encoding categorical features...")

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Define column types
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in X.columns if col not in num_cols]

# Label Encoding for categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

print(f"✓ Encoded {len(cat_cols)} categorical features")

# Step 4: Feature Scaling
print("\n[4/7] Scaling numerical features...")

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print("✓ Numerical features scaled")

# Step 5: Train-Test Split
print("\n[5/7] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Step 6: Model Training
print("\n[6/7] Training ensemble model...")

# Train individual models
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression(max_iter=1000)
clf3 = AdaBoostClassifier()

# Create Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)],
    voting='soft'
)

# Train the model
voting_clf.fit(X_train, y_train)

print("✓ Model trained successfully")

# Step 7: Evaluate Model
print("\n[7/7] Evaluating model performance...")

# Make predictions
y_pred = voting_clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\n{'='*60}")
print("MODEL PERFORMANCE METRICS")
print(f"{'='*60}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {conf_matrix[0][0]}")
print(f"  False Positives: {conf_matrix[0][1]}")
print(f"  False Negatives: {conf_matrix[1][0]}")
print(f"  True Positives:  {conf_matrix[1][1]}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Save the model and preprocessors
print(f"\n{'='*60}")
print("SAVING MODEL AND PREPROCESSORS")
print(f"{'='*60}")

# Save the trained model
with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
print("✓ Model saved: models/churn_model.pkl")

# Save label encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Label encoders saved: models/label_encoders.pkl")

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: models/scaler.pkl")

# Save feature names
feature_names = X.columns.tolist()
with open('models/feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("✓ Feature names saved: models/feature_names.json")

# Save target encoder
with open('models/target_encoder.pkl', 'wb') as f:
    pickle.dump(le_target, f)
print("✓ Target encoder saved: models/target_encoder.pkl")

print(f"\n{'='*60}")
print("✓ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print("\nNext steps:")
print("1. Run 'python app.py' to start the Flask web application")
print("2. Access the application at http://localhost:5000")
print("3. Use the web interface or API endpoints for predictions")
print(f"{'='*60}\n")