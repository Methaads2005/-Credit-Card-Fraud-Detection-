import os
import sys
import numpy as np
import pandas as pd
import joblib

def get_input(prompt, default_val, val_type=float):
    """Safely prompts the user for input with a default value."""
    try:
        user_input = input(f"{prompt} [{default_val}]: ").strip()
        if not user_input:
            return default_val
        return val_type(user_input)
    except ValueError:
        print(f"Invalid input type. Using default value: {default_val}")
        return default_val

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_model.joblib")
    data_path = os.path.join(base_dir, "data", "creditcard.csv")
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Please run 'python src/train.py' first.")
        sys.exit(1)
        
    print("Loading best model pipeline...")
    pipeline = joblib.load(model_path)
    
    print("\n" + "="*50)
    print("      Credit Card Fraud Detection: Manual Input")
    print("="*50)
    print("Please enter the transaction details below.")
    print("Press [Enter] to accept the default baseline value.")
    print("Note: V1-V4 are key PCA components. Rest V5-V28 defaults to 0.0.")
    print("-"*50)
    
    # Prompts
    trans_time = get_input("Transaction Time (seconds since start of day)", 3600.0, float)
    trans_amount = get_input("Transaction Amount ($)", 50.0, float)
    
    # Anonymized key features (typically have the highest correlation)
    v1 = get_input("PCA Feature V1", 0.0, float)
    v2 = get_input("PCA Feature V2", 0.0, float)
    v3 = get_input("PCA Feature V3", 0.0, float)
    v4 = get_input("PCA Feature V4", 0.0, float)
    
    # Build complete input feature dictionary
    input_data = {
        'Time': [trans_time],
        'Amount': [trans_amount],
        'V1': [v1], 'V2': [v2], 'V3': [v3], 'V4': [v4]
    }
    
    # Fill remaining V5-V28 with 0.0 defaults
    for i in range(5, 29):
        input_data[f'V{i}'] = [0.0]
        
    # Convert to DataFrame (ensure columns are ordered exactly as original data)
    # Original order is: Time, V1-V28, Amount
    cols_order = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_df = pd.DataFrame(input_data)[cols_order]
    
    # Extract preprocessing and model from pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    # Run preprocessor
    input_preprocessed = preprocessor.transform(input_df)
    
    # Predict based on model type
    model_name = type(model).__name__
    is_anomaly_detector = "IsolationForest" in model_name
    
    print("\n" + "="*50)
    print("                 PREDICTION RESULT")
    print("="*50)
    print(f"Active Model: {model_name}")
    print("-"*50)
    
    if is_anomaly_detector:
        # Isolation Forest prediction
        pred = model.predict(input_preprocessed)[0] # 1 (normal), -1 (anomaly)
        score = model.score_samples(input_preprocessed)[0]
        
        is_fraud = (pred == -1)
        status = "FRAUDULENT (Anomaly)" if is_fraud else "LEGITIMATE"
        
        print(f"Status: {status}")
        # Scale score for readability: score_samples ranges usually from -0.8 to -0.3
        # -0.8 is highly anomalous, -0.3 is normal
        anomaly_score = -score
        print(f"Anomaly Score: {anomaly_score:.4f} (Higher means more suspicious)")
        
    else:
        # Standard supervised classification
        pred = model.predict(input_preprocessed)[0]
        prob = model.predict_proba(input_preprocessed)[0] # [P(Legit), P(Fraud)]
        
        status = "FRAUDULENT" if pred == 1 else "LEGITIMATE"
        confidence = prob[pred] * 100
        
        print(f"Status: {status}")
        print(f"Confidence Score: {confidence:.2f}%")
        print(f"Probability of Fraud: {prob[1] * 100:.2f}%")
        
    print("="*50)

if __name__ == "__main__":
    main()
