# AI-Driven Control Suite - DC Motor

**Control Theory x Reinforcement Learning for high-performance motor speed regulation**

---

## Project Overview

This project implements an end-to-end AI control stack for an armature-controlled DC motor, where a PPO (Proximal Policy Optimization) agent is trained to replace a classical PID controller for speed regulation.

Key outcomes:

- 2x faster rise time than PID (0.770 s vs 1.450 s)
- 5.5x smoother control voltage transitions (Voltage DeltaV std: 0.087 vs 0.479 V/step)
- Robust operation under sensor noise using noisy observations plus Kalman and Moving-Average filtering

The three required pillars are fully covered:

| Requirement               | Implementation                                                |
| ------------------------- | ------------------------------------------------------------- |
| Regulation de vitesse     | Motor maintains omega = 1.0 rad/s despite electrical dynamics |
| Variateur                 | Voltage saturation, anti-windup, and disturbance handling     |
| Lissage / Echantillonnage | Gaussian encoder noise + Kalman and Moving-Average filters    |

---

## Key Features

- Physics-based Gymnasium environment with coupled electrical and mechanical motor dynamics
- PPO training pipeline using Stable-Baselines3 with normalized observations
- v4 incremental (delta) action-space design that structurally prevents bang-bang switching
- Built-in saturation and anti-windup behavior aligned with industrial constraints
- Signal-processing module with discrete Kalman filter and Moving-Average filter
- Evaluation workflow with side-by-side PID vs PPO metrics and publication-ready plots

---

## 🎬 Demo

▶️ **Watch the full project demo (3 min):**
[LinkedIn Demo Post](https://www.linkedin.com/posts/sadek-dhokar-318342326_reinforcementlearning-controltheory-machinelearning-ugcPost-7442247786714238977-iJFL)

---

## Quick Start

### Installation

```bash
pip install stable-baselines3[extra] gymnasium shimmy torch matplotlib numpy
```

### Usage

1. Train the PPO agent (about 10-15 min on CPU)

```bash
python train_agent.py
```

Outputs:

- ppo_dc_motor_v4.zip
- vec_normalize_v4.pkl
- best_model_v4/best_model.zip

2. Evaluate controllers and generate comparison plots

```bash
python evaluate_and_plot.py
```

Outputs:

- control_comparison.png (4-panel comparison figure)
- lissage_comparison.png (Kalman vs Moving-Average filter comparison)

3. Run filter-only test (no GPU needed)

```bash
python signal_processing.py
```

---

## File Architecture

```text
.
├── motor_env.py          <- Gymnasium environment (plant + sensor noise) [v4]
├── signal_processing.py  <- Kalman filter and Moving-Average filter
├── train_agent.py        <- PPO training loop (Stable-Baselines3) [v4]
├── evaluate_and_plot.py  <- Comparison plots and performance metrics
└── README.md             <- Project documentation
```

---

## Physical Model

Armature-controlled DC motor ODEs (Forward-Euler, dt = 0.01 s):

```text
L * di/dt = V(t) - R*i(t) - K*omega(t)   [Electrical sub-system]
J * domega/dt = K*i(t) - b*omega(t)      [Mechanical sub-system]
```

TP3 parameters:

| Symbol | Value | Unit   | Description                |
| ------ | ----- | ------ | -------------------------- |
| J      | 0.01  | kg.m^2 | Rotor inertia              |
| b      | 0.10  | N.m.s  | Viscous damping            |
| K      | 0.01  | N.m/A  | Electromechanical constant |
| R      | 1.00  | ohm    | Armature resistance        |
| L      | 0.50  | H      | Armature inductance        |
| V_max  | 12.0  | V      | Supply voltage limit       |

Steady-state speed for V = 12 V:

omega_ss = K*V / (R*b + K^2) ~= 1.199 rad/s

Target speed is set to omega\* = 1.0 rad/s.

---

## Main Technical Improvements (v4)

### Delta (Incremental) Action Space

Early versions (v1-v3) used absolute control:

u in [-1, 1] -> V = ((u + 1) / 2) \* V_max

This allowed jumps from 12 V to 0 V in one step, which produced bang-bang behavior that reward shaping alone could not reliably remove.

v4 changes the architecture: the policy outputs a voltage increment Delta u at each step, then the environment integrates it:

```text
u[k] = clip(u[k-1] + MAX_DELTA * Delta u[k], -1, 1)
V[k] = ((u[k] + 1) / 2) * V_max

where MAX_DELTA = 0.05  -> max slew rate = 60 V/s
       Full 0 -> 12 V ramp requires >= 20 steps = 0.2 s
```

Result: high-frequency chattering is physically impossible by construction.

### Observation Space (5D, normalized)

```text
obs[0] = (omega* - omega_noisy) / omega*     normalized speed error
obs[1] = omega_noisy / omega*                normalized current speed
obs[2] = i / i_max                           normalized armature current
obs[3] = integral(e dt) / INTEGRAL_MAX       normalized integral (anti-windup)
obs[4] = u_current                           current voltage level in [-1, 1]
                                                    (required for planning increments)
```

### Action Space

```text
Delta u in [-1, 1] (scaled internally by MAX_DELTA_PER_STEP = 0.05)
```

### Reward Function (v4)

```text
r = -2*e_hat^2 - 0.5*|e_hat|      (L2 + L1 tracking cost)
    -0.02*u^2                      (effort on actual voltage level)
    +2.0*[|e_hat| < 2%]            (tight precision bonus)
    +1.0*[|e_hat| < 5%]            (loose precision bonus)

where e_hat = (omega* - omega) / omega*      normalized error
       u = current normalized voltage level in [-1, 1]
```

The smoothness penalty used in v3/v3.1 is intentionally removed because the delta action-space architecture already enforces smooth control transitions.

### Discrete Kalman Filter

```text
Predict:  x_hat_minus[k] = x_hat[k-1]      (constant-velocity model, A = 1)
           P_minus[k] = P[k-1] + Q

Update:   K_g = P_minus[k] / (P_minus[k] + R_n)
           x_hat[k] = x_hat_minus[k] + K_g * (z[k] - x_hat_minus[k])
           P[k] = (1 - K_g) * P_minus[k]

Tuning:   Q = 1e-5   (process noise variance: speed changes slowly)
           R_n = 4e-4 (measurement noise variance: sigma_noise^2 = 0.02^2)
```

### PID Controller (Discrete Baseline)

```text
Gains: Kp = 5.0, Ki = 12.0, Kd = 0.80, integral_max = 5.0

u[k] = Kp*e[k] + Ki*sum(e*dt) + Kd*Delta e/dt
u[k] = clip(u[k], 0, 12)    [voltage saturation]
```

Gain rationale:

- Original Kp = 8 and Ki = 30 caused about 16% overshoot due to strong proportional kick and integrator windup during saturation.
- Reduced Kp lowers the kick.
- integral_max = 5.0 limits windup accumulation.
- Kd = 0.80 adds damping to reduce ringing.

---

## Performance Comparison

Computed metrics:

- Rise time t_r: time from 10% to 90% of omega\*
- Settling time t_s: first entry into +/-2% band (and remains)
- Overshoot M_p: (omega_max - omega*) / omega* \* 100%
- RMSE: sqrt(mean((omega\* - omega)^2)) over full episode
- Steady-state error: mean |omega\* - omega| over last 10% of episode
- Voltage DeltaV std: std(diff(V(t))) as a smoothness diagnostic

Results (v4, 1M training steps):

| Metric                | PID Controller  | PPO Agent        | Winner                   |
| --------------------- | --------------- | ---------------- | ------------------------ |
| Rise Time (10%->90%)  | 1.450 s         | **0.770 s**      | PPO - 2x faster          |
| Settling Time (+/-2%) | **4.460 s**     | 4.780 s          | PID - slightly faster    |
| Overshoot             | **2.24 %**      | 3.23 %           | Comparable               |
| RMSE                  | 0.331 rad/s     | **0.317 rad/s**  | PPO - slightly lower     |
| Steady-State Error    | **0.004 rad/s** | 0.013 rad/s      | PID - integral guarantee |
| Voltage DeltaV std    | 0.479 V/step    | **0.087 V/step** | PPO - 5.5x smoother      |

The PPO agent converged to a strategy that applies near-full voltage during transient acceleration and then drops near approximately 10 V in steady state. This exploits motor inductance as a natural current limiter and improves transient speed. PID keeps a slight steady-state precision advantage due to its explicit integral action guarantee.

---

## Skills Demonstrated

- Reinforcement Learning for continuous control (PPO, Stable-Baselines3)
- Classical control engineering (PID tuning, anti-windup, saturation handling)
- Physics-based dynamic system modeling (electrical and mechanical coupling)
- Signal processing for noisy measurements (Kalman and Moving-Average filtering)
- Reward shaping and action-space design for stable real-time control behavior
- Performance benchmarking using control-system metrics (rise time, settling time, overshoot, RMSE)
- Reproducible ML experimentation and evaluation workflow

---

## 📝 License

**Copyright (c) 2026 Sadek Dhokar. All Rights Reserved.**

This project is provided for educational and portfolio-demonstration purposes only. **Commercial use, modification, and distribution are strictly prohibited.** For commercial licensing inquiries, enterprise integration, or startup partnerships, please contact me directly at [dhokarsadek.enetcom@gmail.com](mailto:dhokarsadek.enetcom@gmail.com). See the LICENSE file for full details.

---

_Developed for the Industrial Computer Engineering Curriculum - ENET'COM_
