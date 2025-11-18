"""
API Test Script
Quick script to test the prediction API
"""

import requests
import json

# API endpoint
BASE_URL = "http://localhost:5000"

# Sample customer data for testing
sample_customer = {
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0
}

print("="*60)
print("CUSTOMER CHURN PREDICTION - API TEST")
print("="*60)

# Test 1: Health Check
print("\n[Test 1] Health Check")
print("-" * 60)
try:
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✓ Health check passed")
except Exception as e:
    print(f"✗ Health check failed: {str(e)}")

# Test 2: Single Prediction
print("\n[Test 2] Single Customer Prediction")
print("-" * 60)
print("Sample Customer Data:")
print(json.dumps(sample_customer, indent=2))
print()

try:
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample_customer,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Prediction Result:")
        print("-" * 60)
        print(f"Churn: {'YES' if result['churn'] else 'NO'}")
        print(f"Churn Probability: {result['churn_probability']*100:.2f}%")
        print(f"Risk Level: {result['risk_level']}")
        print("✓ Prediction successful")
    else:
        print(f"✗ Prediction failed with status code: {response.status_code}")
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"✗ Prediction failed: {str(e)}")

# Test 3: Batch Prediction
print("\n[Test 3] Batch Prediction (2 customers)")
print("-" * 60)

# Create second customer with different characteristics
sample_customer_2 = sample_customer.copy()
sample_customer_2.update({
    "Contract": "Two year",
    "tenure": 60,
    "MonthlyCharges": 45.5,
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes"
})

batch_data = {
    "customers": [sample_customer, sample_customer_2]
}

try:
    response = requests.post(
        f"{BASE_URL}/api/batch_predict",
        json=batch_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Batch Prediction Results:")
        print("-" * 60)
        for i, pred in enumerate(result['predictions']):
            print(f"\nCustomer {i+1}:")
            print(f"  Churn: {'YES' if pred['churn'] else 'NO'}")
            print(f"  Probability: {pred['churn_probability']*100:.2f}%")
            print(f"  Risk Level: {pred['risk_level']}")
        print("\n✓ Batch prediction successful")
    else:
        print(f"✗ Batch prediction failed with status code: {response.status_code}")
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"✗ Batch prediction failed: {str(e)}")

print("\n" + "="*60)
print("API TEST COMPLETED")
print("="*60 + "\n")