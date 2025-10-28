import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from category_encoders.target_encoder import TargetEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Load data
# df = pd.read_csv("housing.csv") #Full dataset
# df = pd.read_csv("housing_sample2k.csv") #Sample dataset 1
df = pd.read_csv("housing_sample10k.csv") #Sample dataset 2
print(f"Loaded {len(df)} rows")
print(f"Dataset shape: {df.shape}")
print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False))

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
# ADVANCED FEATURE ENGINEERING
# ===============================

# Laundry and parking features
df['has_laundry'] = (df['laundry_options'] != 'none').astype(int)
df['has_parking'] = (df['parking_options'] != 'none').astype(int)

# print("------------\n",df.head())

# Encode categorical variables with frequency encoding for high cardinality
def frequency_encode(df, column):
    freq = df[column].value_counts(normalize=True).to_dict()
    return df[column].map(freq)

encoder = TargetEncoder(cols=['region'])
df['region_encoded'] = encoder.fit_transform(df[['region']], df['price'])


cat_cols = ['type', 'state']
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

# # ===============================
# # MODEL TRAINING WITH MULTIPLE ALGORITHMS
# # ===============================

# # Prepare data
X = df.drop(columns=['price'])  # Drop region but keep region_freq
y = np.log1p(df['price'])  # Log transform target for better distribution

# # Bin target into quantiles for stratification
y_binned = pd.qcut(df['price'], q=10, labels=False, duplicates="drop")

# # Visualize the price distribution before and after log transform
# sns.histplot(df['price'], bins=50, kde=True)

# plt.tight_layout()
# plt.show()


# # Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33, stratify=y_binned) #Sample dataset 2


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


# 1. XGBoost with extensive hyperparameter tuning
# print("Optimizing XGBoost...")
# xgb_params = {
#     'max_depth': [4, 6, 8, 10],
#     'learning_rate': [0.01, 0.03, 0.05, 0.1],
#     'n_estimators': [300, 500, 800, 1000],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [1, 1.5, 2],
#     'min_child_weight': [1, 3, 5]
# }

# xgb_model = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     random_state=33,
#     n_jobs=-1,
#     verbosity=0
# )

# xgb_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=xgb_params,
#     n_iter=50,
#     scoring=rmse_scorer,
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# xgb_search.fit(X_train, y_train)
# xgb_best = xgb_search.best_estimator_
# models_results.append(evaluate_model(xgb_best, X_train, X_test, y_train, y_test, "XGBoost"))

# 2. LightGBM
# print("\nOptimizing LightGBM...")
# lgb_params = {
#     'num_leaves': [31, 50, 100, 150],
#     'learning_rate': [0.01, 0.03, 0.05, 0.1],
#     'n_estimators': [300, 500, 800, 1000],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'reg_alpha': [0, 0.1, 0.5],
#     'reg_lambda': [1, 1.5, 2],
#     'min_child_samples': [20, 50, 100]
# }

# lgb_model = lgb.LGBMRegressor(
#     objective='regression',
#     random_state=33,
#     n_jobs=-1,
#     verbosity=-1
# )

# lgb_search = RandomizedSearchCV(
#     estimator=lgb_model,
#     param_distributions=lgb_params,
#     n_iter=40,
#     scoring=rmse_scorer,
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# lgb_search.fit(X_train, y_train)
# lgb_best = lgb_search.best_estimator_
# models_results.append(evaluate_model(lgb_best, X_train, X_test, y_train, y_test, "LightGBM"))

# 3. CatBoost
# print("\nOptimizing CatBoost...")
# cat_params = {
#     'depth': [6, 8, 10],
#     'learning_rate': [0.01, 0.02, 0.03],
#     'iterations': [1000, 2000, 3000, 5000], 
#     'l2_leaf_reg': [3, 5, 7, 10, 15],
#     'subsample': [0.7, 0.8, 0.9, 1.0],
#     'colsample_bylevel': [0.7, 0.8, 1.0]
# }

# categorical_features = ['type_encoded', 'state_encoded']
# cat_model = cb.CatBoostRegressor(
#     loss_function='RMSE',
#     random_seed=33,
#     cat_features=categorical_features,
#     iterations=5000,
#     learning_rate=0.02,
#     depth=8,
#     l2_leaf_reg=5,  # can be tuned in search
#     subsample=0.9,  # can be tuned
#     colsample_bylevel=0.8,  # can be tuned
#     early_stopping_rounds=200,
#     verbose=False
# )


# cat_search = RandomizedSearchCV(
#     estimator=cat_model,
#     param_distributions=cat_params,
#     n_iter=30,
#     scoring=rmse_scorer,
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# cat_search.fit(X_train, y_train, eval_set=(X_test, y_test))
# cat_best = cat_search.best_estimator_
# print(cat_search.best_params_)
# models_results.append(evaluate_model(cat_best, X_train, X_test, y_train, y_test, "CatBoost"))


# 4. Random Forest
print("\nOptimizing Random Forest...")

# 1️⃣ Define model with fixed params
rf_model = RandomForestRegressor(
    n_estimators=600,
    max_depth=40,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features=0.8,
    criterion='absolute_error',
    random_state=33,
    n_jobs=-1
)

# 2️⃣ Fit the model
rf_model.fit(X_train_scaled, y_train)

# 3️⃣ Evaluate
models_results.append(
    evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")
)

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
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df.head(15)['feature'][::-1], importance_df.head(15)['importance'][::-1])
    plt.title(f'Top 15 Feature Importance ({best_model["name"]})')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


# ===============================
# MODEL PERSISTENCE
# ===============================
print("\nSaving the best model...")

# Save the model
joblib.dump(rf_model, 'random_forest_model.pkl')

# # Load the model later
# rf_loaded = joblib.load('random_forest_model.pkl')

# # Make predictions with the loaded model
# y_pred = rf_loaded.predict(X_test_scaled)

# {'n_estimators': 600, 'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 0.8, 'max_depth': 40, 'criterion': 'absolute_error'}

# Random Forest Results:
# RMSE: $241.59
# MAE: $177.93
# R²: 0.5955