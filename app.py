# pip install -r requirements.txt

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
rf_loaded = joblib.load('pkls/random_forest_model.pkl')
encoder = joblib.load('pkls/target_encoder.pkl')
label_encoders = joblib.load('pkls/label_encoders.pkl')
scaler = joblib.load('pkls/scaler.pkl')

# Load historic data for CAGR calculations
historic_df = pd.read_csv("csvs/state_historic.csv")
historic_df['state'] = historic_df['state'].str.strip().str.lower()

# Get all year columns (2015-2025)
year_columns = [col for col in historic_df.columns if col.isdigit() and 2015 <= int(col) <= 2025]

# Compute state-wise CAGR (using 2020 and 2025 for prediction)
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
    if os.path.exists(os.path.join('static', path)):
        return send_from_directory('static', path)
    elif os.path.exists(path):
        return send_from_directory('.', path)
    else:
        return "File not found", 404

# ===============================
# VALIDATION HELPER FUNCTION
# ===============================
def validate_input(data):
    """Validate that all required fields are provided and not empty"""
    required_fields = {
        'sqft': 'Square Footage',
        'beds': 'Beds',
        'baths': 'Baths',
        'type': 'House Type',
        'state': 'State',
        'city': 'City',
        'lat': 'Latitude',
        'long': 'Longitude'
    }
    
    errors = []
    
    for field, label in required_fields.items():
        value = data.get(field, '').strip() if isinstance(data.get(field), str) else data.get(field)
        
        if value == '' or value is None:
            errors.append(f"{label} is required")
            continue
            
        # Validate numeric fields
        if field in ['sqft', 'beds']:
            try:
                val = int(value)
                if val <= 0:
                    errors.append(f"{label} must be greater than 0")
            except (ValueError, TypeError):
                errors.append(f"{label} must be a valid whole number")
                
        elif field in ['baths', 'lat', 'long']:
            try:
                val = float(value)
                if field == 'baths' and val < 0:
                    errors.append(f"{label} cannot be negative")
            except (ValueError, TypeError):
                errors.append(f"{label} must be a valid number")
    
    return errors

# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        
        # Validate input
        errors = validate_input(data)
        if errors:
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': errors
            }), 400

        # Parse user input - no defaults, values are guaranteed from validation
        user_input = {
            'region': str(data.get('city', '')).lower(),
            'type': str(data.get('type', '')).lower(),
            'sqfeet': int(data.get('sqft')),
            'beds': int(data.get('beds')),
            'baths': float(data.get('baths')),
            'cats_allowed': int(data.get('cats_allowed', 0)),
            'dogs_allowed': int(data.get('dogs_allowed', 0)),
            'smoking_allowed': int(data.get('smoking_allowed', 0)),
            'wheelchair_access': int(data.get('wheelchair_access', 0)),
            'comes_furnished': int(data.get('comes_furnished', 0)),
            'electric_vehicle_charge': int(data.get('electric_vehicle_charge', 0)),
            'has_laundry': int(data.get('has_laundry', 0)),
            'has_parking': int(data.get('has_parking', 0)),
            'lat': float(data.get('lat')),
            'long': float(data.get('long')),
            'state': str(data.get('state', '')).lower()
        }

        print(f"Processing prediction with input: {user_input}")

        # Prepare DataFrame
        user_df = pd.DataFrame([user_input])

        # Encode categorical features
        encoded = encoder.transform(user_df[['region', 'type']])
        user_df = pd.concat([user_df, encoded.add_suffix('_encoded')], axis=1)
        for col, le in label_encoders.items():
            user_df[col + '_encoded'] = le.transform(user_df[col].astype(str))
        user_df = user_df.drop(['region', 'type', 'state'], axis=1)

        # Align columns and scale
        model_features = ['sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
                          'wheelchair_access', 'comes_furnished', 'lat', 'long', 'has_laundry', 'has_parking',
                          'region_encoded', 'type_encoded', 'state_encoded']
        user_df = user_df.reindex(columns=model_features, fill_value=0)
        user_df_scaled = scaler.transform(user_df)

        # Predict 2020 price
        predicted_price_2020 = np.expm1(rf_loaded.predict(user_df_scaled)[0])

        # Predict 2021-2025 using CAGR
        state = user_input['state'].lower()
        cagr = state_growth.get(state, 0.05)
        predicted_prices = {str(year): predicted_price_2020 * ((1 + cagr) ** (year - 2020)) for year in range(2020, 2026)}

        # Prepare historical state averages
        state_historic = historic_df[historic_df['state'] == state]
        historical_prices = {}
        if not state_historic.empty:
            for year in year_columns:
                avg_price = state_historic[year].mean()
                historical_prices[year] = float(avg_price) if not pd.isna(avg_price) else None

        return jsonify({
            'success': True,
            'price_2020': float(predicted_price_2020),
            'price_2025': float(predicted_prices['2025']),
            'predicted_prices': predicted_prices,
            'cagr': float(cagr),
            'historical_prices': historical_prices,
            'state': state.upper()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)