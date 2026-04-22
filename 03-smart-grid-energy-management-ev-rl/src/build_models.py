"""
Smart Grid Energy Management System
Forecasting Models Builder

Builds 4 predictive models based on the synthetic training data:
- Model 1: Price Classification (Low/Medium/High) - KNN, Logistic Regression
- Model 2: Price Regression (€/kWh) - Linear, Ridge, Lasso, RF, GB
- Model 3: Domestic Load Forecasting (kW) - Ridge, RF
- Model 4: EV Availability (0/1) - Logistic Regression

Author: Sadek Dhokar
Date: April 2026
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Ensure reproducability
np.random.seed(42)

def train_and_evaluate_models():
    print("Loading training data...")
    df = pd.read_csv("data/smart_grid_train.csv")
    
    os.makedirs("models", exist_ok=True)
    
    # ---------------------------------------------------------
    # Model 1: Price Classification
    # ---------------------------------------------------------
    print("\n--- Training Model 1: Price Classification ---")
    features_m1 = ["hour", "day_of_week", "month", "temperature", "is_holiday", "is_weekend", "price_lag_1h", "price_lag_24h"]
    X = df[features_m1]
    y = df["price_category"]
    
    # Encode Target
    y_encoded = y.map({"Low": 0, "Medium": 1, "High": 2})
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler_m1 = StandardScaler()
    X_train_scaled = scaler_m1.fit_transform(X_train)
    X_test_scaled = scaler_m1.transform(X_test)
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_preds = knn.predict(X_test_scaled)
    print(f"KNN Accuracy: {accuracy_score(y_test, knn_preds):.4f}")
    
    # Logistic Regression
    lr_cls = LogisticRegression(max_iter=1000)
    lr_cls.fit(X_train_scaled, y_train)
    lr_preds = lr_cls.predict(X_test_scaled)
    print(f"LogReg Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    
    # Save the best model (LogReg in this case for simplicity/interpretability, or KNN if desired. Let's save KNN)
    best_m1 = knn if accuracy_score(y_test, knn_preds) > accuracy_score(y_test, lr_preds) else lr_cls
    joblib.dump(scaler_m1, "models/scaler_m1.pkl")
    joblib.dump(best_m1, "models/model_1_price_cls.pkl")
    
    # ---------------------------------------------------------
    # Model 2: Price Regression
    # ---------------------------------------------------------
    print("\n--- Training Model 2: Price Regression ---")
    features_m2 = ["hour", "day_of_week", "month", "temperature", "is_holiday", "is_weekend", "price_lag_1h", "price_lag_24h"]
    X = df[features_m2]
    y = df["price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_m2 = StandardScaler()
    X_train_scaled = scaler_m2.fit_transform(X_train)
    X_test_scaled = scaler_m2.transform(X_test)
    
    models_m2 = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.01),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_m2_name = None
    best_m2_model = None
    best_r2 = -float("inf")
    
    for name, model in models_m2.items():
        if name in ["Random Forest", "Gradient Boosting"]:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        print(f"{name} -> R2: {r2:.4f}, MSE: {mse:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_m2_name = name
            best_m2_model = model
            
    # Save best regression model (Usually RF or GB)
    joblib.dump(scaler_m2, "models/scaler_m2.pkl")
    joblib.dump({"model": best_m2_model, "needs_scaling": best_m2_name not in ["Random Forest", "Gradient Boosting"]}, "models/model_2_price_reg.pkl")

    # ---------------------------------------------------------
    # Model 3: Load Forecasting
    # ---------------------------------------------------------
    print("\n--- Training Model 3: Domestic Load Forecasting ---")
    features_m3 = ["hour", "day_of_week", "is_weekend", "month", "temperature", "consumption_lag_1h", "consumption_lag_24h"]
    X = df[features_m3]
    y = df["consumption"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ridge
    scaler_m3 = StandardScaler()
    X_train_scaled = scaler_m3.fit_transform(X_train)
    X_test_scaled = scaler_m3.transform(X_test)
    
    ridge = Ridge()
    ridge.fit(X_train_scaled, y_train)
    ridge_preds = ridge.predict(X_test_scaled)
    print(f"Ridge -> R2: {r2_score(y_test, ridge_preds):.4f}, MSE: {mean_squared_error(y_test, ridge_preds):.4f}")
    
    # RF
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print(f"Random Forest -> R2: {r2_score(y_test, rf_preds):.4f}, MSE: {mean_squared_error(y_test, rf_preds):.4f}")
    
    joblib.dump(scaler_m3, "models/scaler_m3.pkl")
    joblib.dump(rf, "models/model_3_load_rf.pkl")
    
    # ---------------------------------------------------------
    # Model 4: EV Availability
    # ---------------------------------------------------------
    print("\n--- Training Model 4: EV Availability ---")
    features_m4 = ["hour", "day_of_week", "is_weekend", "is_holiday", "month"]
    X = df[features_m4]
    y = df["ev_availability"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_cls = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf_cls.fit(X_train, y_train)
    ev_preds = rf_cls.predict(X_test)
    print(f"RF Classifier Accuracy: {accuracy_score(y_test, ev_preds):.4f}")
    
    joblib.dump(rf_cls, "models/model_4_ev_rf.pkl")
    print("\nAll models trained and saved to models/ directory.")

if __name__ == "__main__":
    train_and_evaluate_models()
