"""
Smart Grid Energy Management System
RL Environment Module

This script defines a custom Gymnasium environment for the smart grid agent.

State Space (8 dimensions):
- Current Price (€/kWh)
- Predicted Price Category (0: Low, 1: Medium, 2: High)
- PV Production (kW)
- Load Forecast (kW)
- Stationary Battery SOC (State of Charge, 0-1)
- EV SOC (0-1)
- EV Availability (1: Home, 0: Away)
- Hour of the day (0-23)

Discrete Action Space (6 actions):
0: Charge Stationary Battery (from Grid/PV)
1: Discharge Stationary Battery (to Home)
2: Charge EV (from Grid/PV, if available)
3: V2H from EV (to Home, if available and SOC > 30%)
4: Buy from Grid (satisfy load)
5: Sell to Grid (dump PV/Batteries to grid)

Author: Sadek Dhokar
Date: April 2026
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SmartGridEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Represents the Smart Home Energy System with PV, Battery, and EV.
    """
    def __init__(self, data_df, max_steps=None):
        super(SmartGridEnv, self).__init__()
        
        self.df = data_df.reset_index(drop=True)
        self.n_steps = len(self.df)
        if max_steps:
             self.n_steps = min(self.n_steps, max_steps)
             
        self.current_step = 0
        
        # --- System Parameters ---
        # Stationary Battery
        self.stat_capacity_kwh = 10.0
        self.stat_max_power_kw = 5.0
        self.stat_efficiency = 0.95
        
        # EV Battery
        self.ev_capacity_kwh = 40.0
        self.ev_max_power_kw = 7.0
        self.ev_efficiency = 0.90
        self.ev_min_soc = 0.30 # 30%, which is ~12 kWh (60km autonomy)
        
        # Define Action Space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Define Observation Space
        # [current_price, pred_price_cat, pv, load, stat_soc, ev_soc, ev_avail, hour]
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 2.0, 10.0, 10.0, 1.0, 1.0, 1.0, 23.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Internal State tracking
        self.stat_energy = self.stat_capacity_kwh * 0.5  # Start at 50%
        self.ev_energy = self.ev_capacity_kwh * 0.8     # Start at 80%

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        
        # Map price category string to int if needed
        cat_map = {"Low": 0, "Medium": 1, "High": 2}
        cat_val = row["price_category"]
        if isinstance(cat_val, str):
            cat_val = cat_map[cat_val]
            
        obs = np.array([
            row["price"],
            float(cat_val),
            row["pv_production"],
            row["consumption"],
            self.stat_energy / self.stat_capacity_kwh,
            self.ev_energy / self.ev_capacity_kwh,
            row["ev_availability"],
            row["hour"]
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.stat_energy = self.stat_capacity_kwh * 0.5
        self.ev_energy = self.ev_capacity_kwh * 0.8
        return self._get_obs(), {}

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row["price"]
        pv = row["pv_production"]
        load = row["consumption"]
        ev_avail = row["ev_availability"]
        
        # We process 1 hour per step
        time_delta = 1.0 # hour
        
        # Penalty tracking
        penalty = 0.0
        
        # Drain EV slightly if it is away (simulating driving)
        if ev_avail == 0:
            # Drain ~2 kWh per hour away
            self.ev_energy -= 2.0 * time_delta
            self.ev_energy = max(self.ev_energy, 0)
        
        # --- Energy Balance logic baseline ---
        net_load = load - pv  # positive means we need power, negative means surplus PV
        grid_exchange = 0.0   # positive means buying from grid, negative means selling
        
        # Handle Actions
        if action == 0: # Charge Stationary
            charge_amount = min(self.stat_max_power_kw * time_delta, 
                              (self.stat_capacity_kwh - self.stat_energy))
            # Power comes from PV surplus, if not enough, from grid
            self.stat_energy += charge_amount * self.stat_efficiency
            net_load += charge_amount
            
        elif action == 1: # Discharge Stationary
            discharge_amount = min(self.stat_max_power_kw * time_delta, 
                                 self.stat_energy * self.stat_efficiency)
            self.stat_energy -= discharge_amount / self.stat_efficiency
            # Power goes to satisfy load
            net_load -= discharge_amount
            
        elif action == 2: # Charge EV
            if ev_avail == 1:
                charge_amount = min(self.ev_max_power_kw * time_delta, 
                                  (self.ev_capacity_kwh - self.ev_energy))
                self.ev_energy += charge_amount * self.ev_efficiency
                net_load += charge_amount
            else:
                penalty -= 1.0 # Invalid action penalty
                
        elif action == 3: # V2H from EV
            if ev_avail == 1:
                # Calculate available discharge considering 30% hard constraint
                available_to_discharge = max(0, self.ev_energy - (self.ev_min_soc * self.ev_capacity_kwh))
                discharge_amount = min(self.ev_max_power_kw * time_delta, 
                                     available_to_discharge * self.ev_efficiency)
                
                if discharge_amount <= 0:
                    penalty -= 2.0 # Penalty for trying to discharge below 30%
                else:
                    self.ev_energy -= discharge_amount / self.ev_efficiency
                    net_load -= discharge_amount
                    
                    # Ensure V2H is only done in exceptional high prices
                    cat_map = {"Low": 0, "Medium": 1, "High": 2}
                    cat_val = row["price_category"] if isinstance(row["price_category"], int) else cat_map[row["price_category"]]
                    if cat_val != 2:
                         penalty -= 5.0 # Heavy penalty for non-exceptional V2H usage
            else:
                penalty -= 1.0 # Invalid action penalty
                
        elif action == 4: # Buy from Grid
            # The net_load is just bought from grid. Basically do nothing special with batteries.
            pass
            
        elif action == 5: # Sell to Grid
            # If we want to sell, we could dump the stationary battery to the grid as well
            discharge_amount = min(self.stat_max_power_kw * time_delta, 
                                 self.stat_energy * self.stat_efficiency)
            self.stat_energy -= discharge_amount / self.stat_efficiency
            net_load -= discharge_amount
        
        # After specific actions, handle residual net_load
        if net_load > 0:
            grid_exchange = net_load # Buy from grid
        else:
            grid_exchange = net_load # Sell to grid
            
        # Calculate cost
        # cost = buying_power * price - selling_power * (price * 0.8) # Selling is usually cheaper
        cost = 0.0
        if grid_exchange > 0:
            cost = grid_exchange * price
        else:
            cost = grid_exchange * (price * 0.8) # 80% feed-in tariff
            
        # Reward is negative cost (we want to minimize cost) + penalties
        reward = -cost + penalty
        
        # Enforce EV SOC minimum
        if self.ev_energy < (self.ev_min_soc * self.ev_capacity_kwh) and ev_avail == 1:
            # Huge penalty if it drops below safely while at home, but the logic above prevents V2H from causing it.
            # Driving away might cause it if it started low and was away for a long time.
            reward -= 10.0
            
        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        truncated = False
        info = {
            "cost": cost,
            "grid_exchange": grid_exchange,
            "net_load": net_load,
            "stat_soc": self.stat_energy / self.stat_capacity_kwh,
            "ev_soc": self.ev_energy / self.ev_capacity_kwh,
            "penalty": penalty
        }
        
        # Make sure observations are right
        if not done:
            obs = self._get_obs()
        else:
            obs = self.reset()[0]
            
        return obs, reward, done, truncated, info

    def render(self):
        pass
