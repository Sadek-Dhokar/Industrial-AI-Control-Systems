# AI-Driven Control Suite — DC Motor

## Final Project: Control Theory × Reinforcement Learning

---

### Project Overview

This project builds a complete **AI-Driven Control Suite** for an armature-controlled
DC motor. A PPO (Proximal Policy Optimisation) reinforcement learning agent is trained
to replace a classical PID speed controller. All three required project pillars are
implemented:

| Requirement                   | Implementation                                            |
| ----------------------------- | --------------------------------------------------------- |
| **Régulation de vitesse**     | Motor maintains ω = 1.0 rad/s despite electrical dynamics |
| **Variateur**                 | Voltage saturation, anti-windup, and disturbance handling |
| **Lissage / Échantillonnage** | Gaussian encoder noise + Kalman & Moving-Average filters  |

---

### File Architecture

```
.
├── motor_env.py          ← Gymnasium environment (plant + sensor noise) [v4]
├── signal_processing.py  ← Kalman filter & Moving-Average filter
├── train_agent.py        ← PPO training loop (Stable-Baselines3)       [v4]
├── evaluate_and_plot.py  ← Comparison plots & performance metrics
└── README.md             ← This file
```

---

### Physical Model

Armature-controlled DC motor ODEs (Forward-Euler, dt = 0.01 s):

```
L · di/dt  = V(t) - R·i(t) - K·ω(t)      [Electrical sub-system]
J · dω/dt  = K·i(t) - b·ω(t)             [Mechanical sub-system]
```

**TP3 Parameters:**

| Symbol | Value | Unit  | Description                |
| ------ | ----- | ----- | -------------------------- |
| J      | 0.01  | kg·m² | Rotor inertia              |
| b      | 0.10  | N·m·s | Viscous damping            |
| K      | 0.01  | N·m/A | Electromechanical constant |
| R      | 1.00  | Ω     | Armature resistance        |
| L      | 0.50  | H     | Armature inductance        |
| V_max  | 12.0  | V     | Supply voltage limit       |

**Steady-state speed** (V = 12 V):
ω_ss = K·V / (R·b + K²) ≈ **1.199 rad/s** → Target set to **ω\* = 1.0 rad/s**

---

### Installation

```bash
pip install stable-baselines3[extra] gymnasium shimmy torch matplotlib numpy
```

---

### Usage

## 🎬 Demo

▶️ [Watch the full project demo (3 min)](https://github.com/Sadek-Dhokar/Industrial-IoT-ML-Portfolio/releases/tag/v1.0)

**Step 1 — Train the PPO agent (~10–15 min on CPU)**

```bash
python train_agent.py
```

Outputs: `ppo_dc_motor_v4.zip`, `vec_normalize_v4.pkl`, `best_model_v4/best_model.zip`

**Step 2 — Evaluate and plot**

```bash
python evaluate_and_plot.py
```

Outputs:

- `control_comparison.png` — 4-panel comparison figure
- `lissage_comparison.png` — Kalman vs MA filter demo

**Step 3 — Test filters only (no GPU needed)**

```bash
python signal_processing.py
```

---

### RL Environment Design (v4)

#### Key architectural change: Delta (Incremental) Action Space

Early versions (v1–v3) gave the agent direct/absolute voltage control:
`u ∈ [-1,1] → V = ((u+1)/2) × V_max`. This allowed the agent to switch
from 12V to 0V in a single step, producing **bang-bang control** that
reward-shaping alone could not suppress.

The v4 fix is architectural: the agent now outputs a **voltage increment**
Δu each step, not an absolute voltage level. The environment integrates
the action internally:

```
u[k] = clip(u[k-1] + MAX_DELTA · Δu[k],  -1, 1)
V[k] = ((u[k] + 1) / 2) × V_max

where  MAX_DELTA = 0.05  →  max slew rate = 60 V/s
       Full 0 → 12 V ramp requires ≥ 20 steps = 0.2 s
```

Chattering is **physically impossible** regardless of the reward function.

#### Observation space (5D, normalised)

```
obs[0] = (ω* - ω_noisy) / ω*       normalised speed error
obs[1] = ω_noisy / ω*               normalised current speed
obs[2] = i / i_max                  normalised armature current
obs[3] = ∫e dt / INTEGRAL_MAX       normalised integral (anti-windup)
obs[4] = u_current                  current voltage level ∈ [-1, 1]
                                    (needed so agent can plan increments)
```

#### Action space

```
Δu ∈ [-1, 1]   (scaled internally by MAX_DELTA_PER_STEP = 0.05)
```

#### Reward function (v4)

```
r = − 2·ê²  −  0.5·|ê|          (L2 + L1 tracking cost)
  − 0.02·u²                      (effort on actual voltage level)
  + 2.0·[|ê| < 2%]               (tight precision bonus)
  + 1.0·[|ê| < 5%]               (loose precision bonus)

where  ê = (ω* − ω) / ω*   (normalised error)
       u  = current normalised voltage level ∈ [-1, 1]
```

The smoothness penalty from v3/v3.1 is intentionally removed — it is no
longer needed because the delta action space guarantees smooth output.

---

### Discrete Kalman Filter

```
Predict:   x̂⁻[k] = x̂[k-1]         (constant-velocity model, A = 1)
           P⁻[k]  = P[k-1] + Q

Update:    K_g     = P⁻[k] / (P⁻[k] + R_n)
           x̂[k]   = x̂⁻[k] + K_g · (z[k] - x̂⁻[k])
           P[k]    = (1 - K_g) · P⁻[k]

Tuning:    Q   = 1e-5   (process noise variance — speed changes slowly)
           R_n = 4e-4   (measurement noise variance = σ_noise² = 0.02²)
```

---

### PID Controller (Discrete, Reference Baseline)

```
Gains: Kp = 5.0,  Ki = 12.0,  Kd = 0.80,  integral_max = 5.0

u[k] = Kp·e[k]  +  Ki·∑e·dt  +  Kd·Δe/dt
u[k] = clip(u[k], 0, 12)     [voltage saturation]
```

Rationale for these gains: the original Kp=8 / Ki=30 produced ~16% overshoot
because the large proportional kick combined with integrator windup during the
initial saturation phase. Reducing Kp lowers the proportional kick; the tighter
integral_max=5.0 prevents windup accumulation; Kd=0.80 adds derivative damping
to prevent ringing.

---

### Performance Metrics Computed

- **Rise time** t_r : time from 10 % to 90 % of ω\*
- **Settling time** t_s : first entry into ±2 % band (and stays)
- **Overshoot** M_p : (ω_max - ω*) / ω* × 100 %
- **RMSE** : √(mean((ω\* - ω)²)) over full episode
- **Steady-state error** : mean |ω\* - ω| over last 10 % of episode
- **Voltage ΔV std** : std(diff(V(t))) — smoothness diagnostic

---

### Actual Results (v4 — 1M training steps)

| Metric              | PID Controller  | PPO Agent        | Winner                   |
| ------------------- | --------------- | ---------------- | ------------------------ |
| Rise Time (10%→90%) | 1.450 s         | **0.770 s**      | PPO — 2× faster          |
| Settling Time (±2%) | **4.460 s**     | 4.780 s          | PID — slightly faster    |
| Overshoot           | **2.24 %**      | 3.23 %           | Comparable               |
| RMSE                | 0.331 rad/s     | **0.317 rad/s**  | PPO — slightly lower     |
| Steady-State Error  | **0.004 rad/s** | 0.013 rad/s      | PID — integral guarantee |
| Voltage ΔV std      | 0.479 V/step    | **0.087 V/step** | PPO — 5.5× smoother      |

The PPO agent discovered autonomously that applying full 12V during the
transient (near time-optimal) then dropping to ~10V at steady state is the
optimal strategy. This is faster than the PID because it fully exploits
the motor's inductance as a natural current limiter. The PID's slight edge
in steady-state precision is mathematically guaranteed by the integral term
— the RL agent has no equivalent guarantee.

---

## 📝 License

**Copyright (c) 2026 Sadek Dhokar. All Rights Reserved.**

This project is provided for educational and portfolio-demonstration purposes only. **Commercial use, modification, and distribution are strictly prohibited.** For commercial licensing inquiries, enterprise integration, or startup partnerships, please contact me directly at [dhokarsadek.enetcom@gmail.com](mailto:dhokarsadek.enetcom@gmail.com). See the `LICENSE` file for full details.

---

_Developed for the Industrial Computer Engineering Curriculum — ENET’COM_
