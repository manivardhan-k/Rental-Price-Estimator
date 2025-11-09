from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# ===============================
# LOAD MODELS AND ENCODERS
# ===============================
rf_loaded = joblib.load('random_forest_model.pkl')
encoder = joblib.load('target_encoder.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Load historic data for CAGR calculations
historic_df = pd.read_csv("state_historic.csv")
historic_df['state'] = historic_df['state'].str.strip().str.lower()
historic_df = historic_df[['state', '2020', '2025']]

# Compute state-wise CAGR
state_growth = historic_df.groupby('state')[['2020', '2025']].apply(
    lambda x: ((x['2025'].mean() / x['2020'].mean()) ** (1/5)) - 1
).to_dict()

# ===============================
# SERVE STATIC FILES
# ===============================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_files(path):
    # Check if file exists in static folder
    if os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    # Check if file exists in root directory (for JSON files, etc.)
    elif os.path.exists(path):
        return send_from_directory('.', path)
    else:
        return "File not found", 404

# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        print("Received prediction request")  # Debug log
        data = request.json
        print(f"Request data: {data}")  # Debug log
        
        # Define columns
        int_cols = ['sqfeet', 'beds']
        float_cols = ['baths', 'lat', 'long']
        bool_cols = ['cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access',
                     'electric_vehicle_charge', 'comes_furnished', 'has_laundry', 'has_parking']
        
        # Parse and prepare user input
        user_input = {
            'region': data.get('city', '').lower(),
            'type': data.get('type', '').lower(),
            'sqfeet': int(data.get('sqft', 0)),
            'beds': int(data.get('beds', 0)),
            'baths': float(data.get('baths', 0)),
            'cats_allowed': int(data.get('cats_allowed', 0)),
            'dogs_allowed': int(data.get('dogs_allowed', 0)),
            'smoking_allowed': int(data.get('smoking_allowed', 0)),
            'wheelchair_access': int(data.get('wheelchair_access', 0)),
            'comes_furnished': int(data.get('comes_furnished', 0)),
            'electric_vehicle_charge': int(data.get('electric_vehicle_charge', 0)),
            'has_laundry': int(data.get('has_laundry', 0)),
            'has_parking': int(data.get('has_parking', 0)),
            'lat': float(data.get('lat', 0)),
            'long': float(data.get('long', 0)),
            'state': data.get('state', '').lower()
        }
        
        # Prepare DataFrame
        user_df = pd.DataFrame([user_input])
        
        # Target encoding
        encoded = encoder.transform(user_df[['region', 'type']])
        user_df = pd.concat([user_df, encoded.add_suffix('_encoded')], axis=1)
        
        # Label encoding for state
        for col, le in label_encoders.items():
            user_df[col + '_encoded'] = le.transform(user_df[col].astype(str))
        
        # Drop original categorical columns
        user_df = user_df.drop(['region', 'type', 'state'], axis=1)
        
        # Align columns with model
        model_features = ['sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
                          'wheelchair_access', 'comes_furnished', 'lat', 'long', 'has_laundry', 'has_parking',
                          'region_encoded', 'type_encoded', 'state_encoded']
        
        user_df = user_df.reindex(columns=model_features, fill_value=0)
        
        # Scale features
        user_df_scaled = scaler.transform(user_df)
        
        # Predict 2020 price
        predicted_price_2020 = np.expm1(rf_loaded.predict(user_df_scaled)[0])
        
        # Forecast 2025 price using CAGR
        state = user_input['state'].lower()
        cagr = state_growth.get(state, 0.05)  # default 5% if unknown
        predicted_price_2025 = predicted_price_2020 * ((1 + cagr) ** 5)
        
        return jsonify({
            'success': True,
            'price_2020': float(predicted_price_2020),
            'price_2025': float(predicted_price_2025),
            'cagr': float(cagr)
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Debug log
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)