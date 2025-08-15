#!/usr/bin/env python3
"""
Customer Churn Prediction Demo
Simple script to demonstrate the trained model
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model_and_data():
    """Load the trained model and scaler"""
    try:
        # Load the best model
        model = joblib.load('best_churn_model.pkl')
        print("✅ Model loaded successfully!")
        
        # Load sample data for demonstration
        df = pd.read_csv('customer_churn_dataset.csv')
        print(f"✅ Dataset loaded: {df.shape[0]} customers, {df.shape[1]} features")
        
        return model, df
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")
        return None, None

def predict_churn_sample(model, df, sample_size=5):
    """Make predictions on a sample of customers"""
    print(f"\n🎯 Making predictions on {sample_size} random customers...")
    
    # Select random sample
    sample = df.sample(n=sample_size, random_state=42)
    
    # Prepare features (assuming the same preprocessing as training)
    feature_columns = [col for col in df.columns if col != 'Churn']
    X_sample = sample[feature_columns]
    
    # Make predictions
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]  # Probability of churning
    
    # Display results
    print("\n📊 Prediction Results:")
    print("-" * 60)
    for i, (idx, row) in enumerate(sample.iterrows()):
        churn_status = "🚨 HIGH RISK" if predictions[i] == 1 else "✅ LOW RISK"
        prob = probabilities[i] * 100
        print(f"Customer {i+1}: {churn_status} (Churn Probability: {prob:.1f}%)")
    
    return sample, predictions, probabilities

def show_model_info(model, df):
    """Display information about the model and dataset"""
    print("\n🔍 Model Information:")
    print("-" * 30)
    print(f"Model Type: {type(model).__name__}")
    print(f"Dataset Size: {df.shape[0]:,} customers")
    print(f"Features: {df.shape[1]-1} (excluding target)")
    print(f"Target Distribution:")
    churn_counts = df['Churn'].value_counts()
    print(f"  - Stay (0): {churn_counts[0]:,} customers")
    print(f"  - Leave (1): {churn_counts[1]:,} customers")
    print(f"  - Churn Rate: {churn_counts[1]/len(df)*100:.1f}%")

def main():
    """Main demo function"""
    print("🎯 Customer Churn Prediction Demo")
    print("=" * 40)
    
    # Load model and data
    model, df = load_model_and_data()
    if model is None:
        return
    
    # Show model information
    show_model_info(model, df)
    
    # Make sample predictions
    sample, predictions, probabilities = predict_churn_sample(model, df)
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 To run the full analysis:")
    print("   jupyter notebook customer-churn-prediction-portfolio.ipynb")

if __name__ == "__main__":
    main()
