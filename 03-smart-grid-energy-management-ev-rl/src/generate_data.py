"""
Smart Grid Energy Management System
Data Generation Module

This script generates synthetic hourly data for:
- Energy Prices (€/kWh) and Categories (Low, Medium, High)
- PV Production (kW) based on Temperature and time of day
- Domestic Load Consumption (kW)
- EV Home Availability (Binary)
- External features (Temperature, Day of Week, Is Weekend, Is Holiday, Month)
- Lagged features for forecasting

Author: Sadek Dhokar
Date: April 2026
"""

import pandas as pd
import numpy as np
import os

# Set a consistent random seed
np.random.seed(42)

def generate_synthetic_data(days=365):
    hours = days * 24
    
    # Generate Time Series
    dates = pd.date_range(start="2025-01-01", periods=hours, freq="H")
    df = pd.DataFrame({"datetime": dates})
    
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Synthetic holidays (~10 days)
    random_holidays = np.random.choice(dates.strftime("%Y-%m-%d").unique(), 10, replace=False)
    df["is_holiday"] = df["datetime"].dt.strftime("%Y-%m-%d").isin(random_holidays).astype(int)
    
    # Temperature: Daily seasonality + annual seasonality + noise
    annual_temp = 15 + 10 * np.sin(2 * np.pi * df["datetime"].dt.dayofyear / 365.25 - np.pi/2)
    daily_temp = 5 * np.sin(2 * np.pi * (df["hour"] - 8) / 24)
    df["temperature"] = annual_temp + daily_temp + np.random.normal(0, 2, hours)
    
    # EV Home Availability (0: absent, 1: present)
    # Absent heavily during weekdays 8h-18h (mostly work)
    df["ev_availability"] = 1
    absent_mask = (df["is_weekend"] == 0) & (df["is_holiday"] == 0) & (df["hour"] >= 8) & (df["hour"] < 18)
    df.loc[absent_mask, "ev_availability"] = 0
    # Add a little randomness so he might come home early or leave late sometimes
    df.loc[absent_mask, "ev_availability"] = np.where(np.random.rand(absent_mask.sum()) < 0.05, 1, 0)
    
    # Domestic Load Forecasting (Base shape + morning/evening peaks)
    base_load = 0.5
    morning_peak = 1.0 * np.exp(-0.5 * ((df["hour"] - 7) / 2)**2)
    evening_peak = 1.5 * np.exp(-0.5 * ((df["hour"] - 19) / 2)**2)
    temperature_load = np.where(df["temperature"] < 10, (10 - df["temperature"]) * 0.1, 0)
    temperature_load += np.where(df["temperature"] > 25, (df["temperature"] - 25) * 0.15, 0)
    
    df["consumption"] = base_load + morning_peak + evening_peak + temperature_load
    # Decrease consumption during work hours if EV is absent and no one's home
    df["consumption"] *= np.where(df["ev_availability"] == 0, 0.6, 1.0)
    df["consumption"] += np.abs(np.random.normal(0, 0.2, hours)) 
    
    # PV Production
    # Only produces during daytime (~6 to ~18), peaks around noon
    sun_intensity = np.clip(np.sin(np.pi * (df["hour"] - 6) / 12), 0, 1)
    cloud_cover = np.random.uniform(0.5, 1.0, hours) # Random cloud noise
    efficiency = np.clip(1 - (df["temperature"] - 25) * 0.005, 0.8, 1.0) # Temperature penalty
    df["pv_production"] = 3.0 * sun_intensity * cloud_cover * efficiency  # Max 3kW peak
    
    # Price Regression (€/kWh)
    # Base price follows a curve: higher during evening peaks, low at night
    base_price = 0.15
    price_night = np.where((df["hour"] >= 0) & (df["hour"] < 6), -0.05, 0)
    price_evening = np.where((df["hour"] >= 17) & (df["hour"] <= 21), 0.10, 0)
    price_noise = np.random.normal(0, 0.02, hours)
    df["price"] = np.clip(base_price + price_night + price_evening + price_noise, 0.05, 0.50)
    
    # Add extreme high prices for V2H testing events
    spike_mask = np.random.rand(hours) < 0.02
    df.loc[spike_mask, "price"] *= np.random.uniform(2.0, 3.5)
    df.loc[spike_mask, "price"] = np.clip(df.loc[spike_mask, "price"], 0.05, 1.0)
    
    # Price Classification
    q33 = df["price"].quantile(0.33)
    q66 = df["price"].quantile(0.66)
    
    def categorize_price(p):
        if p < q33: return "Low"
        if p < q66: return "Medium"
        return "High"
        
    df["price_category"] = df["price"].apply(categorize_price)
    
    # Create Lags
    df["price_lag_1h"] = df["price"].shift(1).fillna(method="bfill")
    df["price_lag_24h"] = df["price"].shift(24).fillna(method="bfill")
    df["consumption_lag_1h"] = df["consumption"].shift(1).fillna(method="bfill")
    df["consumption_lag_24h"] = df["consumption"].shift(24).fillna(method="bfill")
    
    return df

def main():
    print("Generating 1 year of synthetic data for the Smart Grid RL project...")
    df = generate_synthetic_data(days=365)
    
    # Save datasets
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/smart_grid_data.csv", index=False)
    
    # We will split the last 30 days as a pure simulation testing benchmark
    train_df = df.iloc[:-30*24]
    test_df = df.iloc[-30*24:]
    
    train_df.to_csv("data/smart_grid_train.csv", index=False)
    test_df.to_csv("data/smart_grid_sim_test.csv", index=False)
    
    print(f"Total Rows: {len(df)}")
    print("Saved to data/ folder successfully.")

if __name__ == "__main__":
    main()
