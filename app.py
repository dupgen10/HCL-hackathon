# Customer Churn Prediction MVP - Flask Application
# Complete web application for hackathon submission

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json

app = Flask(__name__)
# Load model when the app starts (important for Render)


# Global variables for model and encoders
model = None
label_encoders = {}
scaler = None
feature_names = []

def load_model_and_preprocessors():
    """Load trained model and preprocessing objects"""
    global model, label_encoders, scaler, feature_names
    
    try:
        # Load the trained model
        with open('models/churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load label encoders
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load scaler
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open('models/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        print("✓ Model and preprocessors loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False
load_model_and_preprocessors()
@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make churn prediction"""
    try:
        # Get form data
        data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Preprocess the input
        processed_data = preprocess_input(input_df)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Prepare response
        result = {
            'churn': bool(prediction),
            'churn_probability': float(probability[1]),
            'no_churn_probability': float(probability[0]),
            'risk_level': get_risk_level(probability[1])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def preprocess_input(df):
    """Preprocess input data to match training format"""
    
    # Numerical columns to scale
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Categorical columns for label encoding
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Convert string values to appropriate types
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Apply label encoding to categorical columns
    for col in cat_cols:
        if col in df.columns and col in label_encoders:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except:
                # Handle unseen labels
                df[col] = 0
    
    # Scale numerical features
    if scaler is not None:
        df[num_cols] = scaler.transform(df[num_cols])
    
    # Ensure correct column order
    df = df[feature_names]
    
    return df

def get_risk_level(probability):
    """Determine risk level based on churn probability"""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Medium'
    else:
        return 'High'

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple customers"""
    try:
        data = request.get_json()
        
        if 'customers' not in data:
            return jsonify({'error': 'Missing customers data'}), 400
        
        customers = data['customers']
        input_df = pd.DataFrame(customers)
        
        # Preprocess all inputs
        processed_data = preprocess_input(input_df)
        
        # Make predictions
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)
        
        # Prepare results
        results = []
        for i in range(len(predictions)):
            results.append({
                'customer_id': i,
                'churn': bool(predictions[i]),
                'churn_probability': float(probabilities[i][1]),
                'risk_level': get_risk_level(probabilities[i][1])
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Load model and preprocessors
    if not load_model_and_preprocessors():
        print("⚠ Warning: Model not loaded. Please train the model first by running train_model.py")
    
    # Run the Flask app
    print("\n" + "="*50)
    print("🚀 Customer Churn Prediction MVP")
    print("="*50)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)