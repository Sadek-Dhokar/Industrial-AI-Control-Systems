"""
Smart Grid Energy Management System
Simulation and Evaluation Script

Simulates a 30-day testing benchmark evaluating three strategies:
1. Baseline (Rule-based naive logic: Charge battery immediately when PV > Load)
2. ML-Only (Forecast-driven logic)
3. ML + RL (PPO Agent informed by state/forecasts)

Author: Sadek Dhokar
Date: April 2026
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from src.env import SmartGridEnv

def run_baseline_strategy(env, test_df):
    """
    Run a naive rule-based strategy on the test dataset.
    """
    obs, info = env.reset()
    total_cost = 0.0
    costs = []
    stat_socs = []
    ev_socs = []
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        pv = row["pv_production"]
        load = row["consumption"]
        net_load = load - pv
        ev_avail = row["ev_availability"]
        
        # Naive Strategy Rules
        if net_load < 0: # PV Surplus
            # Charge Stationary Battery
            if env.stat_energy < env.stat_capacity_kwh:
                action = 0 # Charge
            elif ev_avail == 1 and env.ev_energy < env.ev_capacity_kwh:
                action = 2 # Charge EV
            else:
                action = 5 # Sell to Grid
        else: # Need Power
            if env.stat_energy > 0:
                action = 1 # Discharge Stationary
            else:
                action = 4 # Buy from Grid
                
        obs, reward, done, truncated, info = env.step(action)
        
        total_cost += info["cost"]
        costs.append(info["cost"])
        stat_socs.append(info["stat_soc"])
        ev_socs.append(info["ev_soc"])
        
    return total_cost, costs, stat_socs, ev_socs

def run_ml_only_strategy(env, test_df):
    """
    Run an ML-forecast-driven strategy on the test dataset.
    (Placeholder for a slightly smarter programmatic approach based on ML models)
    """
    obs, info = env.reset()
    total_cost = 0.0
    costs = []
    
    # Load Models
    price_cls = joblib.load("models/model_1_price_cls.pkl")
    scaler_m1 = joblib.load("models/scaler_m1.pkl")
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        
        # We would theoretically use our models to predict here,
        # but for simulation simplicity we'll just check the real category
        # to approximate a perfect forecaster driving heuristics:
        
        cat_map = {"Low": 0, "Medium": 1, "High": 2}
        cat = row["price_category"] if isinstance(row["price_category"], int) else cat_map[row["price_category"]]
        
        # Rule based on price category
        if cat == 0: # Low Price -> Charge Batteries
            if env.ev_energy < env.ev_capacity_kwh * 0.8 and row["ev_availability"] == 1:
                action = 2
            elif env.stat_energy < env.stat_capacity_kwh:
                action = 0
            else:
                action = 4
        elif cat == 2: # High Price -> Discharge / V2H
            if row["ev_availability"] == 1 and env.ev_energy > env.ev_capacity_kwh * 0.4:
                action = 3 # V2H
            elif env.stat_energy > 0:
                action = 1 # Discharge Stat
            else:
                action = 5 # Sell
        else: # Medium Price -> Naive behavior
             action = 0 if (row["pv_production"] > row["consumption"]) else 4
             
        obs, reward, done, truncated, info = env.step(action)
        
        total_cost += info["cost"]
        costs.append(info["cost"])
        
    return total_cost, costs

def run_rl_strategy(env, model, test_df):
    """
    Run the trained PPO Agent strategy on the test dataset.
    """
    obs, info = env.reset()
    total_cost = 0.0
    costs = []
    actions = []
    
    for i in range(len(test_df)):
        action, _states = model.predict(obs, deterministic=True)
        # Handle scalar actions properly
        action_val = action.item() if isinstance(action, np.ndarray) else action
            
        obs, reward, done, truncated, info = env.step(action_val)
        
        total_cost += info["cost"]
        costs.append(info["cost"])
        actions.append(action_val)
        
    return total_cost, costs, actions

def main():
    print("Loading test data (last 30 days)...")
    test_df = pd.read_csv("data/smart_grid_sim_test.csv")
    
    # Re-instantiate Environment with Test Data
    env_baseline = SmartGridEnv(test_df)
    env_ml = SmartGridEnv(test_df)
    env_rl = SmartGridEnv(test_df)
    
    # Run Baseline
    print("\nRunning Baseline Strategy Simulation...")
    base_cost, base_costs, _, _ = run_baseline_strategy(env_baseline, test_df)
    print(f"Total Baseline Cost: €{base_cost:.2f}")
    
    # Run ML Only
    print("\nRunning ML-Only Strategy Simulation...")
    ml_cost, ml_costs = run_ml_only_strategy(env_ml, test_df)
    print(f"Total ML-Only Cost: €{ml_cost:.2f}")
    
    # Run RL
    print("\nRunning ML+RL Strategy Simulation...")
    try:
        model = PPO.load("models/ppo_smart_grid_agent")
        rl_cost, rl_costs, rl_actions = run_rl_strategy(env_rl, model, test_df)
        print(f"Total ML+RL Cost: €{rl_cost:.2f}")
    except Exception as e:
        print(f"Could not load RL Agent. Ensure you have run train_agent.py first.\nError: {e}")
        return

    # --- Plotting and Comparisons ---
    os.makedirs("figures", exist_ok=True)
    
    # Cumulative Cost over 30 Days Plot
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(base_costs), label="Baseline (Rule-based)", color="red", linestyle="--")
    plt.plot(np.cumsum(ml_costs), label="ML-Only (Forecasts Heuristics)", color="orange")
    plt.plot(np.cumsum(rl_costs), label="ML + RL Additive Agent", color="blue", linewidth=2)
    plt.title("Cumulative Electricity Cost over 30 Days", fontsize=14)
    plt.xlabel("Hour of Simulation", fontsize=12)
    plt.ylabel("Cumulative Cost (€)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/cumulative_cost_comparison_rl_benchmark.png")
    
    # Savings metrics
    base_savings = 0.0
    ml_savings = (base_cost - ml_cost) / base_cost * 100
    rl_savings = (base_cost - rl_cost) / base_cost * 100
    
    print("\n--- Simulation Summary ---")
    print(f"Baseline total: €{base_cost:.2f}")
    print(f"ML-Only total:  €{ml_cost:.2f} ({ml_savings:.1f}% savings)")
    print(f"RL Agent total: €{rl_cost:.2f} ({rl_savings:.1f}% savings)")
    print("Cost comparison plot saved to figures/cumulative_cost_comparison_rl_benchmark.png")
    
if __name__ == "__main__":
    main()
