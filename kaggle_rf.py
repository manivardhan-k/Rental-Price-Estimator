import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from category_encoders.target_encoder import TargetEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# https://www.kaggle.com/datasets/austinreese/usa-housing-listings

# Load data
# df = pd.read_csv("csvs/housing.csv") #Full dataset
df = pd.read_csv("csvs/housing_sample2k.csv") #Sample dataset 1
# df = pd.read_csv("csvs/housing_sample10k.csv") #Sample dataset 2
# df = pd.read_csv("csvs/housing_sample20k.csv") #Sample dataset 3
# df = pd.read_csv("csvs/housing_sample50k.csv") #Sample dataset 4

# print(df.head())
# print(df.columns)

# print(f"Loaded {len(df)} rows")
# print(f"Dataset shape: {df.shape}")
# print("\nMissing values:")
# print(df.isnull().sum().sort_values(ascending=False))

# sns.stripplot(x=df['price'])
# plt.show()

# ===============================
# ENHANCED DATA PREPROCESSING
# ===============================

# Identify outliers in price using IQR method
def remove_outliers_iqr(df, column, factor=1.25):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove price & sqfeet outliers
print(f"Before outlier removal: {len(df)} rows")
df = remove_outliers_iqr(df, 'price', factor=1.5)
df = remove_outliers_iqr(df, 'sqfeet', factor=2)
print(f"After outlier removal: {len(df)} rows")

# sns.stripplot(x=df['price'])
# sns.stripplot(x=df['sqfeet'])
# plt.show()

# Drop url columns
df = df.drop(columns= df.filter(regex='url$').columns.tolist())

# Enhanced feature engineering
numeric_cols = ['price', 'sqfeet', 'beds', 'baths']
bool_cols = ['cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access',
             'electric_vehicle_charge', 'comes_furnished']

# Better imputation for numeric columns
for col in numeric_cols:
    if col != 'price':
        # Use median for skewed distributions
        df[col] = df[col].fillna(df[col].median())

# Boolean columns
for col in bool_cols:
    df[col] = df[col].fillna(0).astype(int)

# Categorical columns
df['type'] = df['type'].fillna('Other')
df['laundry_options'] = df['laundry_options'].fillna('none')
df['parking_options'] = df['parking_options'].fillna('none')

# ===============================
# ADDING HISTORIC DATA
# ===============================

# Load the historical data
# historic_df = pd.read_csv("csvs/state_historic.csv")

# Normalize state codes to lowercase for consistency
# historic_df['state'] = historic_df['state'].str.strip().str.lower()
# df['state'] = df['state'].str.strip().str.lower()

# Identify numeric/price columns (assuming they are year columns)
# price_cols = [col for col in historic_df.columns if col.isdigit()]

# Compute engineered features
# historic_df['price_2025'] = historic_df['2025']
# historic_df['price_growth_5y'] = (historic_df['2025'] - historic_df['2020']) / historic_df['2020']
# historic_df['avg_annual_growth'] = ((historic_df['2025'] / historic_df['2020']) ** (1/5)) - 1
# historic_df['volatility'] = historic_df[price_cols].std(axis=1)
# historic_df['recent_trend'] = (historic_df['2025'] - historic_df['2024']) / historic_df['2024']

# Keep only relevant columns
# historic_features = historic_df[['state', 'price_2025']] #'price_growth_5y', 'avg_annual_growth', 
                                # 'volatility', 'recent_trend']]

# Merge into your main df
# df = df.merge(historic_features, how='left', on='state')

# df['price_2025'] = df['price_2025'] * 0.8 + np.random.normal(0, df['price_2025'].std() * 0.05)

# print("------------\n",df.head())


# ===============================
# ADVANCED FEATURE ENGINEERING
# ===============================

# Laundry and parking features
df['has_laundry'] = (df['laundry_options'] != 'none').astype(int)
df['has_parking'] = (df['parking_options'] != 'none').astype(int)

# print("------------\n",df.head())

# Encode categorical variables
encoder = TargetEncoder(cols=['region', 'type'])
encoded = encoder.fit_transform(df[['region', 'type']], df['price'])
df = pd.concat([df, encoded.add_suffix('_encoded')], axis=1)

cat_cols = ['state']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# print("------------\n",df.head())
# sns.stripplot(x=df['price'])
# plt.show()

# Drop unnecessary columns
drop_cols = ['id', 'description', 'laundry_options', 'parking_options', 'region', 'state', 'sqfeet_bins', 'beds_category', 'type', 
             'sqfeet_bins_encoded', 'electric_vehicle_charge']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# print("------------\n",df.head())

print(f"Final feature count: {df.shape[1] - 1}")
print(f"Features: {[col for col in df.columns if col != 'price']}")

df = df.dropna()
# print(df.head())

# ===============================
# MODEL TRAINING WITH MULTIPLE ALGORITHMS
# ===============================

# Prepare data
X = df.drop(columns=['price'])  # Drop region but keep region_freq
y = np.log1p(df['price'])  # Log transform target for better distribution

# Bin target into quantiles for stratification
y_binned = pd.qcut(df['price'], q=10, labels=False, duplicates="drop")

# Visualize the price distribution before and after log transform
# sns.histplot(df['price'], bins=50, kde=True)

# plt.tight_layout()
# plt.show()


# # Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=33, stratify=y_binned) #Sample dataset 2


X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# # Scale features for algorithms that need it - Not required for catboost
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# HYPERPARAMETER OPTIMIZATION
# ===============================

def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    
    # Convert back from log space
    y_te_orig = np.expm1(y_te)
    preds_orig = np.expm1(preds)
    
    rmse = np.sqrt(mean_squared_error(y_te_orig, preds_orig))
    mae = mean_absolute_error(y_te_orig, preds_orig)
    r2 = r2_score(y_te_orig, preds_orig)
    
    print(f"\n{model_name} Results:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.4f}")
    
    return {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2, 'name': model_name}

# RMSE scorer for log-transformed target
def rmse_log(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.expm1(y_true), np.expm1(y_pred)))

rmse_scorer = make_scorer(rmse_log, greater_is_better=False)

models_results = []

import time

start = time.time()

# 4. Random Forest
print("\nOptimizing Random Forest...")

# Random Forest hyperparameter grid
rf_params = {
    'n_estimators': [100],
    'max_depth': [40],
    'min_samples_split': [10],
    'min_samples_leaf': [4],
    'max_features': [0.8],
    'criterion': ['absolute_error']
}

rf_model = RandomForestRegressor(random_state=33, n_jobs=-1)

rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_params,
    n_iter=1,
    scoring=rmse_scorer,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
print(rf_search.best_params_)

# 3️⃣ Evaluate
models_results.append(
    evaluate_model(rf_best, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")
)

end = time.time()

print(f"Training took {(end - start)/60:.2f} minutes")

# ===============================
# RESULTS SUMMARY
# ===============================
print("\n" + "🏆" + "="*50)
print("🎉 FINAL RESULTS SUMMARY")
print("🏆" + "="*50)

all_results = models_results
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('rmse').reset_index(drop=True)

print(results_df[['name', 'rmse', 'mae', 'r2']].round(4).to_string(index=False))

# Best model
best_model = min(models_results, key=lambda x: x['rmse'])
print(f"\n🥇 Best single model: {best_model['name']}")


# ===============================
# FEATURE IMPORTANCE ANALYSIS
# ===============================
print("\nAnalyzing feature importance...")

# Use the best performing model for feature importance
best_single_model = best_model['model']
if hasattr(best_single_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': best_single_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 features ({best_model['name']}):")
    print(importance_df.to_string(index=False))

    # Plot feature importance
    # plt.figure(figsize=(10, 8))
    # plt.barh(importance_df.head(15)['feature'][::-1], importance_df.head(15)['importance'][::-1])
    # plt.title(f'Top 15 Feature Importance ({best_model["name"]})')
    # plt.xlabel('Importance')
    # plt.tight_layout()
    # plt.show()

# ===============================
# MODEL PERSISTENCE
# ===============================
# print("\nSaving the best model...")

# Save the model
# # # # # # # # never run this # # # joblib.dump(best_model['model'], 'pkls/random_forest_model.pkl')
# joblib.dump(encoder, 'pkls/target_encoder.pkl')
# joblib.dump(label_encoders, 'pkls/label_encoders.pkl')
# joblib.dump(scaler, 'pkls/scaler.pkl')

# # Load the model later
# rf_loaded = joblib.load('pkls/random_forest_model.pkl')

# # Make predictions with the loaded model
# y_pred = rf_loaded.predict(X_test_scaled)

# # Convert from log space to original price scale
# y_te_orig = np.expm1(y_test)
# preds_orig = np.expm1(y_pred)

# # Compute metrics in original scale
# rmse = np.sqrt(mean_squared_error(y_te_orig, preds_orig))
# mae = mean_absolute_error(y_te_orig, preds_orig)
# r2 = r2_score(y_te_orig, preds_orig)

# print(f"RMSE: ${rmse:,.2f}")
# print(f"MAE: ${mae:,.2f}")
# print(f"R²: {r2:.4f}")




# {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 0.8, 'max_depth': 40, 'criterion': 'absolute_error'}

# Random Forest Results:
# RMSE: $241.59
# MAE: $177.93
# R²: 0.5955

# Target encoded region
#          name     rmse      mae     r2
# Random Forest 228.0427 153.7749 0.6614

# Target encoded region and type
#          name     rmse      mae     r2
# Random Forest 225.7749 152.6099 0.6681

# Test 0.15
#          name     rmse      mae     r2
# Random Forest 223.0775 151.5975 0.6713

# Sample 20k
#          name     rmse      mae     r2
# Random Forest 219.3287 141.0897 0.6824

# Sample 50k
# Training took 76.77 minutes
#          name     rmse      mae     r2
# Random Forest 199.3254 119.2751 0.7431

# Sample 50k - 100 n_estimaters
# Training took 12.87 minutes
#          name     rmse      mae     r2
# Random Forest 200.3836 120.2867 0.7403

# Sample 385k - 100 n_estimaters
# Should take ~10.5 hrs
# Training took 752.65 minutes
        #  name     rmse     mae     r2
# Random Forest 150.5183 73.0027 0.8541
