"""
motor_env.py
============
Custom Gymnasium Environment: DC Motor Speed Control (v4)
=========================================================

ARCHITECTURE CHANGE in v4 — Incremental (Delta) Action Space
-------------------------------------------------------------
All previous versions used an ABSOLUTE voltage action: the agent directly
outputs the target voltage u[k] ∈ [-1,1] → V ∈ [0, V_MAX].

Problem: an absolute action space makes bang-bang control trivially easy.
The agent can switch from 12V to 0V in a single step, and reward shaping
(smoothness penalty) cannot reliably prevent this because:
  • The penalty must be calibrated against the precision bonus, and PPO's
    high-variance gradient estimates make this balance fragile.
  • The agent simply finds a switching amplitude small enough that the
    penalty is offset by the tracking gain.

Solution — Incremental (delta) action space:
  • The agent outputs Δu[k] ∈ [-1, 1] each step.
  • The actual normalised voltage is: u[k] = clip(u[k-1] + α·Δu[k], -1, 1)
    where α = MAX_DELTA_PER_STEP = 0.05 (max change per step).
  • This limits the voltage slew rate to α × V_MAX / dt = 0.05×12/0.01 = 60 V/s.
  • A full 0→12V ramp takes at minimum 1/α = 20 steps = 0.2 s (physical).
  • Bang-bang is now physically impossible: the agent CANNOT switch from
    12V to 0V in one step regardless of reward coefficients.

Reward (v4 — simplified, no smoothness term needed):
    r = −2·ê²  −  0.5·|ê|   (L2+L1 tracking — removes local optima)
      −  0.02·u²              (effort on actual voltage, not delta)
      + 2.0·[|ê|<2%]          (tight precision bonus)
      + 1.0·[|ê|<5%]          (loose precision bonus)

The smoothness penalty is removed entirely because it is no longer needed
— smooth control is guaranteed by the action space itself.

Observation Space (5-D):
    obs[0] = (ω* − ω_noisy) / ω*    normalised speed error
    obs[1] = ω_noisy / ω*            normalised speed
    obs[2] = i / i_max               normalised current
    obs[3] = ∫e dt / INTEGRAL_MAX    normalised integral
    obs[4] = u_current               current voltage level ∈ [-1, 1]
             (agent needs to know where it is to plan increments)

Physical Model (unchanged — TP3 parameters):
    L·di/dt = V(t) - R·i(t) - K·ω(t)
    J·dω/dt = K·i(t) - b·ω(t)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (TP3 — unchanged)
# ---------------------------------------------------------------------------
J        = 0.01
b_damp   = 0.10
K_motor  = 0.01
R_motor  = 1.00
L_motor  = 0.50
V_MAX    = 12.0

I_MAX        = V_MAX / R_motor   # 12 A
OMEGA_TARGET = 1.0               # rad/s
OMEGA_MAX    = K_motor * V_MAX / (R_motor * b_damp + K_motor**2)

DT           = 0.01
MAX_STEPS    = 500
INTEGRAL_MAX = 10.0
NOISE_SIGMA  = 0.02

# Max voltage change per step (normalised).
# 0.05 → max slew = 0.05 × 12V / 0.01s = 60 V/s
# Full ramp 0→12V takes ≥ 20 steps = 0.2 s.
MAX_DELTA_PER_STEP = 0.05


class DCMotorEnv(gym.Env):
    """
    Gymnasium environment for DC motor speed-regulation (v4).

    Key difference from v1-v3: action = voltage INCREMENT Δu, not absolute u.
    The environment integrates the action to obtain actual voltage.

    Observation: 5-D  [err, speed, current, integral, current_voltage]
    Action:      1-D  Δu ∈ [-1, 1]  (scaled by MAX_DELTA_PER_STEP internally)
    """

    metadata = {"render_modes": []}

    def __init__(self, noise_sigma: float = NOISE_SIGMA,
                 target_speed: float = OMEGA_TARGET,
                 render_mode=None):
        super().__init__()

        self.noise_sigma  = noise_sigma
        self.target_speed = target_speed
        self.render_mode  = render_mode

        # Agent outputs a normalised delta; the env scales it by MAX_DELTA_PER_STEP
        self.action_space = spaces.Box(
            low  = np.array([-1.0], dtype=np.float32),
            high = np.array([ 1.0], dtype=np.float32),
            dtype= np.float32
        )

        # 5-D observation: 4 physics states + current voltage level
        obs_low  = np.array([-5.0, -1.0, -1.0, -5.0, -1.0], dtype=np.float32)
        obs_high = np.array([ 5.0,  5.0,  1.0,  5.0,  1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Internal state
        self.current      = 0.0
        self.omega        = 0.0
        self.integral     = 0.0
        self.omega_noisy  = 0.0
        self._u_current   = -1.0   # starts at 0V (u=-1 → V=0)
        self._step_count  = 0

    def _get_obs(self) -> np.ndarray:
        noise            = self.np_random.normal(0.0, self.noise_sigma)
        self.omega_noisy = self.omega + noise

        error         = self.target_speed - self.omega_noisy
        norm_error    = error / self.target_speed
        norm_speed    = self.omega_noisy / self.target_speed
        norm_current  = self.current / I_MAX
        norm_integral = self.integral / INTEGRAL_MAX

        return np.array(
            [norm_error, norm_speed, norm_current,
             norm_integral, self._u_current],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current     = 0.0
        self.omega       = 0.0
        self.integral    = 0.0
        self.omega_noisy = 0.0
        self._u_current  = -1.0   # 0 V at start
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # 1. Compute new voltage from delta action
        #    delta is scaled by MAX_DELTA_PER_STEP so the agent's output range
        #    [-1, 1] maps to a change of [-MAX_DELTA, +MAX_DELTA] per step.
        delta   = float(np.clip(action[0], -1.0, 1.0)) * MAX_DELTA_PER_STEP
        self._u_current = float(np.clip(self._u_current + delta, -1.0, 1.0))
        voltage = ((self._u_current + 1.0) / 2.0) * V_MAX   # V ∈ [0, V_MAX]

        # 2. Forward-Euler integration
        di_dt     = (voltage - R_motor * self.current
                     - K_motor * self.omega) / L_motor
        domega_dt = (K_motor * self.current - b_damp * self.omega) / J

        self.current += di_dt     * DT
        self.omega   += domega_dt * DT

        self.current = np.clip(self.current, 0.0, I_MAX)
        self.omega   = max(self.omega, 0.0)

        # 3. Integral state
        error          = self.target_speed - self.omega
        self.integral += error * DT
        self.integral  = np.clip(self.integral, -INTEGRAL_MAX, INTEGRAL_MAX)

        self._step_count += 1

        # 4. Reward (v4 — no smoothness term needed)
        norm_error = error / self.target_speed

        tracking_cost = 2.0 * norm_error**2 + 0.5 * abs(norm_error)
        effort_cost   = 0.02 * self._u_current**2

        if abs(norm_error) < 0.02:
            precision_bonus = 2.0
        elif abs(norm_error) < 0.05:
            precision_bonus = 1.0
        else:
            precision_bonus = 0.0

        reward = float(-tracking_cost - effort_cost + precision_bonus)

        # 5. Observation
        obs = self._get_obs()

        terminated = False
        truncated  = (self._step_count >= MAX_STEPS)

        info = {
            "omega"      : self.omega,
            "omega_noisy": self.omega_noisy,
            "current"    : self.current,
            "voltage"    : voltage,
            "error"      : error,
            "step"       : self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        voltage = ((self._u_current + 1.0) / 2.0) * V_MAX
        print(f"[Step {self._step_count:4d}] "
              f"ω={self.omega:.4f} rad/s  "
              f"V={voltage:.2f} V  "
              f"err={self.target_speed - self.omega:.4f}")
