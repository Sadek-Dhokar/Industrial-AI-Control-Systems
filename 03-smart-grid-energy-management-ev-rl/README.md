# Smart Grid Energy Management System with EV and Reinforcement Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Gymnasium-000000?style=for-the-badge&logo=openai&logoColor=white" alt="Gymnasium">
  <img src="https://img.shields.io/badge/Reinforcement_Learning-FF6F00?style=for-the-badge" alt="RL">
</div>

## Project Overview

This project implements an intelligent, AI-driven **Home Energy Management System (HEMS)**. It simulates a modern smart home equipped with:
- **Solar Photovoltaic (PV) Panels**
- **Stationary Battery Storage (10 kWh)**
- **Electric Vehicle (EV) with Vehicle-to-Home (V2H) capabilities (40 kWh)**

The goal of the system is to **minimize household electricity costs** and **maximize solar self-consumption** under a dynamic energy pricing scheme. To achieve this, the project combines traditional **Machine Learning (ML)** for time-series forecasting (predicting prices, solar generation, and household load) with a deep **Reinforcement Learning (RL)** agent that makes optimal hourly decisions on how to route energy.

**Business Value:** This system demonstrates how AI can be leveraged in the energy sector to optimize smart grids, reduce consumer energy bills, and safely orchestrate V2H technology without leaving the user stranded without EV range (hard constraint enforced: EV state-of-charge never drops below 30%).

## Key Features & Technologies

### 1. Machine Learning Forecasting (Scikit-Learn)
Before making decisions, the system predicts the future. We trained multiple supervised learning models to forecast key metrics:
- **Price Classification (Low/Medium/High):** K-Nearest Neighbors (KNN), Logistic Regression.
- **Price Regression (€/kWh):** Ridge, Lasso, Random Forest, Gradient Boosting.
- **Domestic Load Forecasting (kW):** Ridge Regression, Random Forest.
- **EV Availability Prediction:** Random Forest Classifier to predict when the EV is plugged in at home vs. being used for commuting.

### 2. Deep Reinforcement Learning Agent (Stable-Baselines3)
The core decision-maker is a **Proximal Policy Optimization (PPO)** agent trained in a custom **OpenAI Gymnasium** environment. 
- **State Space (Continuous 8D):** Current Price, Predicted Price Tier, PV Production, Current Load, Stationary Battery SOC, EV SOC, EV Availability, and Hour of the Day.
- **Action Space (Discrete 6 Actions):** 
  1. Charge Stationary Battery
  2. Discharge Stationary Battery
  3. Charge EV
  4. Vehicle-to-Home (V2H) (Discharge EV to power the house)
  5. Buy energy from Grid
  6. Sell energy to Grid

### 3. Comprehensive Hourly Simulation
The project includes a 30-day untouched test simulation to benchmark three distinct strategies:
* **Baseline (Rule-Based):** A naive approach that charges batteries only when PV production exceeds load.
* **ML-Only (Forecast Heuristics):** A programmatic approach that uses ML forecasts to buy grid power during "Low" prices and discharge batteries during "High" prices.
* **ML + RL Agent:** The fully autonomous PPO Agent that dynamically learns the best actions to minimize costs while respecting strict battery degradation and EV range constraints.

## Repository Structure

```text
03-smart-grid-energy-management-ev-rl/
├── data/                  # Generated synthetic datasets (1-year hourly data)
├── models/                # Saved ML/RL models
├── figures/               # Output plots comparing strategies (costs vs. time)
├── src/
│   ├── build_models.py    # Trains ML predictive models (Linear, RF, etc.)
│   ├── env.py             # Custom Gymnasium RL environment for the Smart Home
│   ├── generate_data.py   # Synthesizes 1-year hourly load/price/PV/EV profiles
│   ├── simulate.py        # Runs 30-day simulations to benchmark the 3 strategies
│   └── train_agent.py     # PPO agent training script (100k timesteps)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## How to Run

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate the synthetic dataset:**
   ```bash
   python src/generate_data.py
   ```
3. **Train the ML Forecasting Models:**
   ```bash
   python src/build_models.py
   ```
4. **Train the PPO Reinforcement Learning Agent:**
   ```bash
   python src/train_agent.py
   ```
5. **Simulate and Compare Strategies:**
   ```bash
   python src/simulate.py
   ```

## Outcomes & Results
The comprehensive hourly simulation highlights how the **RL + ML** approach significantly outperforms baseline heuristics. The RL agent successfully learns to shift energy consumption to off-peak hours, intelligently utilizes V2H by dumping battery power only during exceptional high-price spikes, and guarantees the EV is always ready for the morning commute.
