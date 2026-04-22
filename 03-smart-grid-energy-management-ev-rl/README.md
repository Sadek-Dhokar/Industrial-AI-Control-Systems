# Smart Grid Energy Management System with EV and Reinforcement Learning Decision Agent

## Project overview
**Student:** Sadek Dhokar – GII Semester 2, ENET’Com  
**Course:** Électronique de Puissance  
**Professor:** Moez Ghariani  
**Date:** April 2026

This project implements an intelligent Energy Management System (EMS) for a smart home equipped with a photovoltaic (PV) array, a stationary battery, and an Electric Vehicle (EV). It combines traditional Machine Learning forecasting models with a deep Reinforcement Learning (RL) agent.

The objective is to minimize electricity costs while maximizing self-consumption of solar energy and respecting battery and EV constraints. Specifically, the EV acts as a flexible load and possible energy source (V2H), bounded by a hard requirement to maintain at least 30% SOC (60 km reserve autonomy, approx. 12 kWh).

## Key Features
1. **Machine Learning Forecasters:**
   - **Model 1: Price Classification** (Low, Medium, High) using KNN & Logistic Regression.
   - **Model 2: Price Regression** (€/kWh) using Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting.
   - **Model 3: Load Forecasting** (kW) predicting domestic consumption using Ridge Regression and Random Forest.
   - **Model 4: EV Availability Prediction**, simulating weekday absence (8h–18h) and weekend/night presence.
   
2. **Deep Reinforcement Learning Agent:**
   - Proximal Policy Optimization (PPO) via `Stable-Baselines3`.
   - **6 Discrete Actions:**
     1. Charge Stationary Battery
     2. Discharge Stationary Battery
     3. Charge EV
     4. V2H (Discharge EV to Home)
     5. Buy from Grid
     6. Sell to Grid
   - **State Space:**
     Current Price, Predicted Price, PV Production, Load Forecast, Stationary Battery SOC, EV SOC, EV Availability, Hour of the day.
   - **Constraints:**
     V2H is only allowed during exceptional high price peaks. The EV SOC is hard-constrained to never drop below 30%.

3. **Hourly Simulation & Benchmarking:**
   - Full 30-day simulation of the home grid utilizing generated synthetic data.
   - Comparison of 3 strategies:
     - Baseline (Rule-based naive strategy)
     - ML-Only (Optimization driven by forecasts only)
     - ML+RL (PPO Agent informed by forecasts)

## Repository Structure
```text
03-smart-grid-energy-management-ev-rl/
├── data/                  # Generated synthetic datasets
├── models/                # Saved ML/RL models
├── figures/               # Output plots comparing strategies
├── src/
│   ├── build_models.py    # ML forecasting models (Linear, RF, etc.)
│   ├── env.py             # Custom Gymnasium RL environment
│   ├── generate_data.py   # Synthesizes 30-day hourly load/price/PV/EV profiles
│   ├── simulate.py        # Runs 30-day simulations to benchmark the 3 strategies
│   └── train_agent.py     # PPO agent training script
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Generate base data: `python src/generate_data.py`
3. Train ML Forecasting Models: `python src/build_models.py`
4. Train the PPO RL agent: `python src/train_agent.py`
5. Simulate and compare strategies: `python src/simulate.py`

## Outcomes
The comprehensive hourly simulation highlights how an RL agent combined with localized weather/load forecasting can drastically reduce energy costs and intelligently shift V2H utilization to only the most expensive time blocks, all while ensuring the EV is ready for daily commuting.
