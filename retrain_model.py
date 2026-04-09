import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json

print("Loading data...")

fabric_df = pd.read_csv('data/fabric_properties.csv')
event_df = pd.read_csv('data/event_properties.csv')
climate_df = pd.read_csv('data/climate_zones.csv')
city_df = pd.read_csv('data/city_mappings.csv')
main_df = pd.read_csv('data/main_training_dataset.csv')

print(f"Dataset loaded: {len(main_df)} rows")
print(f"Columns: {main_df.columns.tolist()}")

# Fix missing precipitation
main_df['precipitation'] = main_df['precipitation'].fillna('Low')

# Define features
categorical_features = [
    'fabric_name', 'fabric_category', 'event_name', 'climate_zone',
    'season', 'setting', 'humidity', 'precipitation'
]

numerical_features = [
    'breathability', 'warmth', 'formality', 'water_resistance',
    'event_formality_level', 'duration_hours', 'activity_level',
    'weather_exposure', 'temp_min', 'temp_max', 'temp_avg'
]

# Encode categorical features
print("\nEncoding categorical features...")
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    main_df[f'{col}_encoded'] = le.fit_transform(main_df[col])
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} categories")

# Create feature columns
feature_columns = [f'{col}_encoded' for col in categorical_features] + numerical_features

# Prepare X and y
X = main_df[feature_columns]
y = main_df['overall_recommendation_score']

print(f"\nFeatures: {len(feature_columns)}")
print(f"Samples: {len(X)}")
print(f"Target range: {y.min():.1f} to {y.max():.1f}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\nTraining Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)
print("Model trained!")

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  MAE: {mae:.4f}")
print(f"  R²:  {r2:.4f} ({r2*100:.2f}%)")

# Save model files
print("\nSaving model files...")

joblib.dump(model, 'model/gradient_boosting_model_JOBLIB.pkl')
print("  Saved: gradient_boosting_model_JOBLIB.pkl")

joblib.dump(label_encoders, 'model/label_encoders_JOBLIB.pkl')
print("  Saved: label_encoders_JOBLIB.pkl")

feature_info = {
    'feature_columns': feature_columns,
    'categorical_features': categorical_features,
    'numerical_features': numerical_features,
    'n_features': len(feature_columns)
}

with open('model/feature_info_JOBLIB.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print("  Saved: feature_info_JOBLIB.json")

print("\nAll model files saved!")
print("Ready to run the app!")