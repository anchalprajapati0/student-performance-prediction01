import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Dataset/student-por[1].csv")

print("Dataset Shape:", df.shape)

# Encode categorical columns and save encoders
encoders = {}

for column in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    encoders[column] = le
# Features and Target
X = df.drop("G3", axis=1)
print(X.columns.tolist())
y = df["G3"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
# ===========================
# Linear Regression
# ===========================
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print("\nLinear Regression")
print("R2 Score:", r2_score(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))

# ===========================
# Random Forest
# ===========================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\nRandom Forest")
print("R2 Score:", r2_score(y_test, rf_pred))
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))

# ===========================
# XGBoost
# ===========================
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print("\nXGBoost")
print("R2 Score:", r2_score(y_test, xgb_pred))
print("MAE:", mean_absolute_error(y_test, xgb_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))

# ===========================
# Save Model
# ===========================
# Create Model folder if it doesn't exist
os.makedirs("Model", exist_ok=True)

# Save model, scaler and encoders
joblib.dump(xgb, "Model/student_model.pkl")
joblib.dump(scaler, "Model/scaler.pkl")
joblib.dump(encoders, "Model/encoders.pkl")

print("\n✅ Model Saved")
print("✅ Scaler Saved")
print("✅ Encoders Saved")