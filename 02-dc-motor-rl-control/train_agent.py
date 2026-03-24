"""
train_agent.py
==============
PPO Agent Training — DC Motor Speed Control (v4)
================================================

Change log:
  v1  original      — absolute action, β=0.05 → 25% undershoot
  v2  Gemini        — path fix only, same broken reward
  v3  Claude        — L1+L2 tracking, β=0.005 → undershoot fixed, chattering
  v3.1 Claude       — smoothness penalty γ=0.15/0.80 → chattering persists
  v4  Claude        — DELTA action space → smooth control guaranteed by design

Why delta actions fix chattering permanently:
  MAX_DELTA_PER_STEP = 0.05 limits slew to 60 V/s.
  Switching 12V→0V takes ≥ 20 steps → impossible to chatter at 100 Hz.
  No reward tuning required — the physics prevents it.

Training is slightly harder (the agent must learn to integrate its own
actions) but 1M steps is sufficient. If the agent undershoots at 1M steps,
extend to 1.5M.

Save paths: ppo_dc_motor_v4.zip, vec_normalize_v4.pkl
"""

import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from motor_env import DCMotorEnv

TOTAL_TIMESTEPS  = 1_000_000
N_ENVS           = 4
LOG_DIR          = "./logs_v4/"
BEST_MODEL_DIR   = "./best_model_v4/"
CHECKPOINT_DIR   = "./checkpoints_v4/"
MODEL_SAVE_PATH  = "ppo_dc_motor_v4"
VECNORM_SAVE     = "vec_normalize_v4.pkl"

os.makedirs(LOG_DIR,        exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class TrainingProgressCallback(BaseCallback):
    PRINT_FREQ = 25_000

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
        if self.num_timesteps % self.PRINT_FREQ == 0 and self._episode_rewards:
            mean_rew = np.mean(self._episode_rewards[-100:])
            print(f"  [Step {self.num_timesteps:>7d}]  "
                  f"Mean reward (last 100 ep): {mean_rew:+.3f}")
        return True


def train():
    print("=" * 60)
    print("  DC Motor PPO Training (v4) — Delta Action Space")
    print("=" * 60)
    print(f"  MAX_DELTA_PER_STEP = 0.05  (max slew 60 V/s)")
    print(f"  Obs: 5-D  [err, speed, current, integral, voltage_level]")
    print(f"  Action: Δu ∈ [-1,1] (scaled by 0.05 internally)\n")

    def make_env():
        return Monitor(DCMotorEnv(noise_sigma=0.02))

    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, gamma=0.99)

    eval_env = VecNormalize(
        make_vec_env(make_env, n_envs=1),
        norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.99,
    )

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 128,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.005,   # slightly higher: delta actions need more exploration
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
        tensorboard_log = LOG_DIR,
        verbose         = 0,
    )

    print(f"Policy: 5 → 128 → 128 → 1  (actor & critic)")
    print(f"Training for {TOTAL_TIMESTEPS:,} steps on {N_ENVS} envs …\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [
            TrainingProgressCallback(),
            EvalCallback(eval_env, best_model_save_path=BEST_MODEL_DIR,
                         log_path=LOG_DIR, eval_freq=20_000 // N_ENVS,
                         n_eval_episodes=10, deterministic=True, verbose=0),
            CheckpointCallback(save_freq=100_000 // N_ENVS,
                               save_path=CHECKPOINT_DIR,
                               name_prefix="ppo_v4"),
        ],
        tb_log_name         = "PPO_DC_Motor_v4",
        reset_num_timesteps = True,
    )

    model.save(MODEL_SAVE_PATH)
    vec_env.save(VECNORM_SAVE)

    print("\n" + "=" * 60)
    print(f"  Final model  → {MODEL_SAVE_PATH}.zip")
    print(f"  Best model   → {BEST_MODEL_DIR}best_model.zip")
    print(f"  VecNormalize → {VECNORM_SAVE}")
    print("=" * 60)

    # Sanity check
    print("\nSanity-check episode …")
    obs    = vec_env.reset()
    speeds, voltages = [], []

    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = vec_env.step(action)
        speeds.append(info[0].get("omega", 0.0))
        voltages.append(info[0].get("voltage", 0.0))

    print(f"  Final speed  : {speeds[-1]:.4f} rad/s  (target 1.0)")
    print(f"  Peak speed   : {max(speeds):.4f} rad/s  "
          f"(overshoot {max(0.0,(max(speeds)-1.0)*100):.1f}%)")
    # Voltage smoothness check: std of diff should be small
    v_diff_std = float(np.std(np.diff(voltages)))
    print(f"  Voltage Δ std: {v_diff_std:.4f} V/step  "
          f"({'smooth ✓' if v_diff_std < 0.5 else 'still chattering ✗'})")

    vec_env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    train()
