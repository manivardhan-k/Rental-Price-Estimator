import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODELS AND ENCODERS
# ===============================
rf_loaded = joblib.load('pkls/random_forest_model.pkl')
encoder = joblib.load('pkls/target_encoder.pkl')
label_encoders = joblib.load('pkls/label_encoders.pkl')
scaler = joblib.load('pkls/scaler.pkl')

# ===============================
# USER INPUT
# ===============================
features = ['sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed',
            'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished', 'lat', 'long',
            'has_laundry', 'has_parking', 'region', 'type', 'state']


int_cols = ['sqfeet', 'beds']
float_cols = ['baths', 'lat', 'long']
bool_cols = ['cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access',
             'electric_vehicle_charge', 'comes_furnished', 'has_laundry', 'has_parking']

# user_input = {}
# print("Please enter the property details:")
# for feature in features:
#     x = input(f"{feature}: ").strip()
#     if feature in int_cols:
#         x = int(x)
#     elif feature in float_cols:
#         x = float(x)
#     elif feature in bool_cols:
#         x = 0 if x.lower() in ['none', '0', 'false', 'no'] else 1
#     user_input[feature] = x

# Sample user input
user_input = {
    'region': 'flint',
    'type': 'apartment',
    'sqfeet': 700,
    'beds': 1,
    'baths': 1.0,
    'cats_allowed': 1,
    'dogs_allowed': 1,
    'smoking_allowed': 1,
    'wheelchair_access': 0,
    'comes_furnished': 0,
    'electric_vehicle_charge': 0,
    'has_laundry': 1,
    'has_parking': 1,
    'lat': 42.9435,
    'long': -83.6072,
    'state': 'mi'
}

'''
boston,
2485, price - 1973
apartment,
1000,
2,
2.0,
1,
1,
0,
0,
0,
0,
1,
1,
42.2093,
-70.9963,
ma
'''

# ===============================
# PREPARE DATAFRAME
# ===============================
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

# ===============================
# PREDICT 2020 PRICE
# ===============================
predicted_price_2020 = np.expm1(rf_loaded.predict(user_df_scaled)[0])
print(f"💰 Estimated 2020 price: ${predicted_price_2020:,.2f}")

# ===============================
# FORECAST 2025 PRICE
# ===============================
historic_df = pd.read_csv("csvs/state_historic.csv")
historic_df['state'] = historic_df['state'].str.strip().str.lower()
historic_df = historic_df[['state', '2020', '2025']]

# Compute state-wise CAGR
state_growth = historic_df.groupby('state')[['2020', '2025']].apply(
    lambda x: ((x['2025'].mean() / x['2020'].mean()) ** (1/5)) - 1
).to_dict()

# Apply CAGR to predict 2025 price
state = user_input['state'].lower()
cagr = state_growth.get(state, 0.05)  # default 5% if unknown
predicted_price_2025 = predicted_price_2020 * ((1 + cagr) ** 5)
print(f"💰 Estimated 2025 price: ${predicted_price_2025:,.2f}")
