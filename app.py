from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Constants
MODEL_PATH = 'model.joblib'
SCALER_PATH = 'scaler.joblib'
FEATURES_PATH = 'feature_columns.joblib'

# Helper to load model and scaler
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, scaler, features
    return None, None, None

def Format_Input(data, features):
    # Create DF from single input row
    df = pd.DataFrame([data])
    
    # Apply log transformations as in the training
    df['total_rooms'] = np.log(df['total_rooms'] + 1)
    df['total_bedrooms'] = np.log(df['total_bedrooms'] + 1)
    df['population'] = np.log(df['population'] + 1)
    df['households'] = np.log(df['households'] + 1)
    
    # One-hot encoding manual (handle missing keys)
    ocean_categories = ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    for cat in ocean_categories:
        df[cat] = 1 if data.get('ocean_proximity') == cat else 0
        
    df = df.drop(['ocean_proximity'], axis=1)
    
    # Extra features
    df['bedroom_radio'] = df['total_bedrooms'] / df['total_rooms']
    df['household_rooms'] = df['total_rooms'] / df['households']
    
    # Ensure columns match training (order and names)
    df = df[features]
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, scaler, features = load_model()
        if model is None:
            return jsonify({'error': 'Model not trained yet. Please run train_model.py first.'}), 400
        
        data = request.json
        # Convert numeric values
        for key in data:
            if key != 'ocean_proximity':
                data[key] = float(data[key])
        
        formatted_df = Format_Input(data, features)
        scaled_data = scaler.transform(formatted_df)
        prediction = model.predict(scaled_data)[0]
        
        return jsonify({
            'prediction': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
