"""
evaluate_and_plot.py
====================
Evaluation & Visualisation — AI vs. Classical PID Control (v4)
================================================================

Changes from v3.1:
  • Loads v4 model (delta action space) with fallback to v3.1/v3/v2/v1.
  • PPO simulation updated: the env now takes delta actions internally,
    so the simulation loop is unchanged — the environment handles it.
  • Added voltage smoothness metric (Δ std) to the metrics table so the
    improvement over bang-bang is quantitatively visible.
  • Annotation positions cleaned up further.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from motor_env import DCMotorEnv, DT, V_MAX, OMEGA_TARGET, MAX_STEPS
from signal_processing import KalmanFilter1D, MovingAverageFilter

PPO_AVAILABLE = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    PPO_AVAILABLE = True
except ImportError:
    print("[WARN] stable-baselines3 not found — PPO trace will be skipped.")

matplotlib.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 12,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 10,
    "figure.dpi"      : 150,
    "lines.linewidth" : 1.8,
    "axes.grid"       : True,
    "grid.alpha"      : 0.35,
})

COLORS = {
    "pid_filt"  : "#C0392B",
    "ppo_filt"  : "#1A5276",
    "setpoint"  : "#27AE60",
    "noise"     : "#BDC3C7",
    "pid_raw"   : "#E74C3C",
    "ma_filter" : "#E67E22",
}


# ===========================================================================
# Metrics
# ===========================================================================
def compute_metrics(time_vec, speed_vec, voltage_vec=None,
                    target=OMEGA_TARGET):
    n = len(time_vec)
    lo, hi  = 0.10 * target, 0.90 * target
    idx_10  = np.argmax(speed_vec >= lo)
    idx_90  = np.argmax(speed_vec >= hi)
    rise_time = (float(time_vec[idx_90] - time_vec[idx_10])
                 if idx_90 > idx_10 else float("nan"))

    peak      = np.max(speed_vec)
    overshoot = max(0.0, (peak - target) / target * 100.0)

    band    = 0.02 * target
    in_band = np.abs(speed_vec - target) <= band
    settled_idx = 0
    for k in range(n - 1, -1, -1):
        if not in_band[k]:
            settled_idx = k + 1
            break
    settled_idx = min(settled_idx, n - 1)

    rmse   = float(np.sqrt(np.mean((speed_vec - target) ** 2)))
    tail   = speed_vec[int(0.9 * n):]
    ss_err = float(np.mean(np.abs(tail - target)))

    v_smooth = float(np.std(np.diff(voltage_vec))) if voltage_vec is not None else float("nan")

    return {
        "rise_time"         : rise_time,
        "settling_time"     : float(time_vec[settled_idx]),
        "overshoot_pct"     : overshoot,
        "rmse"              : rmse,
        "steady_state_error": ss_err,
        "voltage_delta_std" : v_smooth,
    }


# ===========================================================================
# PID controller (retuned v3 gains — unchanged)
# ===========================================================================
class DiscretePID:
    def __init__(self, Kp, Ki, Kd, dt=DT,
                 u_min=0.0, u_max=V_MAX, integral_max=5.0):
        self.Kp = Kp; self.Ki = Ki; self.Kd = Kd
        self.dt = dt; self.u_min = u_min; self.u_max = u_max
        self.integral_max = integral_max
        self._integral = 0.0; self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0; self._prev_error = 0.0

    def compute(self, setpoint, measurement):
        error = setpoint - measurement
        P = self.Kp * error
        self._integral = np.clip(
            self._integral + error * self.dt,
            -self.integral_max, self.integral_max)
        I = self.Ki * self._integral
        D = self.Kd * (error - self._prev_error) / self.dt
        self._prev_error = error
        return float(np.clip(P + I + D, self.u_min, self.u_max))


# ===========================================================================
# Simulations
# ===========================================================================
def run_pid_simulation(Kp=5.0, Ki=12.0, Kd=0.80, use_kalman=True):
    env = DCMotorEnv(noise_sigma=0.02)
    pid = DiscretePID(Kp=Kp, Ki=Ki, Kd=Kd)
    kf  = KalmanFilter1D(Q=1e-5, R_noise=4e-4)
    obs, _ = env.reset(seed=0)
    pid.reset(); kf.reset()

    logs = {k: [] for k in
            ["time","omega_true","omega_noisy","omega_filtered",
             "voltage","error","current"]}

    for step in range(MAX_STEPS):
        omega_filt = kf.update(env.omega_noisy) if use_kalman else env.omega_noisy
        voltage    = pid.compute(OMEGA_TARGET, omega_filt)
        # PID outputs absolute voltage — convert to delta action for v4 env
        # The env integrates internally, so we must match its internal state.
        # Easiest: bypass the delta mechanism by directly setting _u_current.
        u_norm = 2.0 * voltage / V_MAX - 1.0
        env._u_current = float(np.clip(u_norm, -1.0, 1.0))
        # Dummy delta that will be overridden by the clamp logic:
        # We pass action=0 so env computes delta=0, keeping _u_current as set.
        action = np.array([0.0], dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)

        logs["time"].append(step * DT)
        logs["omega_true"].append(info["omega"])
        logs["omega_noisy"].append(info["omega_noisy"])
        logs["omega_filtered"].append(omega_filt)
        logs["voltage"].append(info["voltage"])
        logs["error"].append(info["error"])
        logs["current"].append(info["current"])

        if terminated or truncated:
            break

    env.close()
    return {k: np.array(v) for k, v in logs.items()}


def run_ppo_simulation(model_path="ppo_dc_motor_v4",
                       vecnorm_path="vec_normalize_v4.pkl",
                       use_kalman=True):
    if not PPO_AVAILABLE:
        return None

    candidates = [
        (model_path + ".zip",                            vecnorm_path),
        (os.path.join("best_model_v4","best_model.zip"), vecnorm_path),
        ("ppo_dc_motor_v3_1.zip",                        "vec_normalize_v3_1.pkl"),
        (os.path.join("best_model_v3_1","best_model.zip"),"vec_normalize_v3_1.pkl"),
        ("ppo_dc_motor_v3.zip",                          "vec_normalize_v3.pkl"),
        (os.path.join("best_model_v3","best_model.zip"), "vec_normalize_v3.pkl"),
        ("ppo_dc_motor_v2.zip",                          "vec_normalize_v2.pkl"),
        ("ppo_dc_motor.zip",                             "vec_normalize.pkl"),
    ]

    actual_model = actual_vecnorm = None
    for mp, vp in candidates:
        if os.path.exists(mp):
            actual_model, actual_vecnorm = mp, vp
            print(f"  [PPO] Model: {actual_model}")
            break

    if actual_model is None:
        print("[WARN] No PPO model found. Run train_agent.py first.")
        return None

    def _make():
        return Monitor(DCMotorEnv(noise_sigma=0.02))

    vec_env = DummyVecEnv([_make])
    if actual_vecnorm and os.path.exists(actual_vecnorm):
        vec_env = VecNormalize.load(actual_vecnorm, vec_env)
        vec_env.training = False; vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, gamma=0.99, training=False)

    model   = PPO.load(actual_model, env=vec_env)
    kf      = KalmanFilter1D(Q=1e-5, R_noise=4e-4); kf.reset()
    obs     = vec_env.reset()
    raw_env = vec_env.envs[0].unwrapped

    logs = {k: [] for k in
            ["time","omega_true","omega_noisy","omega_filtered",
             "voltage","error","current"]}

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = vec_env.step(action)
        raw_info   = info[0]
        omega_filt = kf.update(raw_info.get("omega_noisy", raw_env.omega_noisy))

        logs["time"].append(step * DT)
        logs["omega_true"].append(raw_info.get("omega",       raw_env.omega))
        logs["omega_noisy"].append(raw_info.get("omega_noisy", raw_env.omega_noisy))
        logs["omega_filtered"].append(omega_filt)
        logs["voltage"].append(raw_info.get("voltage",  0.0))
        logs["error"].append(raw_info.get("error",    0.0))
        logs["current"].append(raw_info.get("current", 0.0))

        if done[0]:
            break

    vec_env.close()
    return {k: np.array(v) for k, v in logs.items()}


# ===========================================================================
# Metrics table (with voltage smoothness)
# ===========================================================================
def print_metrics_table(pid_data, ppo_data):
    m_pid = compute_metrics(pid_data["time"], pid_data["omega_filtered"],
                            pid_data["voltage"])
    headers = ["Metric", "PID Controller"]
    rows    = [
        ["Rise Time (10%→90%)",   f"{m_pid['rise_time']:.3f} s"],
        ["Settling Time (±2%)",   f"{m_pid['settling_time']:.3f} s"],
        ["Overshoot",             f"{m_pid['overshoot_pct']:.2f} %"],
        ["RMSE",                  f"{m_pid['rmse']:.5f} rad/s"],
        ["Steady-State Error",    f"{m_pid['steady_state_error']:.5f} rad/s"],
        ["Voltage ΔV std (smooth)",f"{m_pid['voltage_delta_std']:.4f} V/step"],
    ]
    if ppo_data is not None:
        m_ppo = compute_metrics(ppo_data["time"], ppo_data["omega_filtered"],
                                ppo_data["voltage"])
        headers.append("PPO Agent")
        vals = [
            f"{m_ppo['rise_time']:.3f} s",
            f"{m_ppo['settling_time']:.3f} s",
            f"{m_ppo['overshoot_pct']:.2f} %",
            f"{m_ppo['rmse']:.5f} rad/s",
            f"{m_ppo['steady_state_error']:.5f} rad/s",
            f"{m_ppo['voltage_delta_std']:.4f} V/step",
        ]
        for row, v in zip(rows, vals):
            row.append(v)

    col_w = 30
    sep = "+" + "+".join(["-"*col_w]*len(headers)) + "+"
    fmt = "|" + "|".join([f" {{:<{col_w-1}}}"]*len(headers)) + "|"
    print("\n" + sep)
    print(fmt.format(*headers))
    print(sep.replace("-","="))
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ===========================================================================
# Plots
# ===========================================================================
def plot_comparison(pid_data, ppo_data, save_path="control_comparison.png"):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "AI-Driven Control Suite — DC Motor Step Response\n"
        "PPO Agent (v4 — smooth delta control)  vs.  Classical PID  (TP3 Parameters)",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    t_pid = pid_data["time"]
    t_end = float(t_pid[-1])

    # Panel 1 — speed
    ax1.axhline(OMEGA_TARGET, color=COLORS["setpoint"], lw=1.5, ls="--",
                label=f"Set-point  ω* = {OMEGA_TARGET} rad/s")
    ax1.axhspan(0.98*OMEGA_TARGET, 1.02*OMEGA_TARGET,
                alpha=0.12, color=COLORS["setpoint"], label="±2 % band")
    ax1.plot(t_pid, pid_data["omega_filtered"],
             color=COLORS["pid_filt"], label="PID (Kalman-filtered)")
    if ppo_data is not None:
        ax1.plot(ppo_data["time"], ppo_data["omega_filtered"],
                 color=COLORS["ppo_filt"], label="PPO Agent (Kalman-filtered)")
    ax1.set_title("Speed Step Response (Régulation de Vitesse)")
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Angular velocity  ω [rad/s]")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_xlim(0, t_end); ax1.set_ylim(-0.05, 1.35*OMEGA_TARGET)

    m_pid  = compute_metrics(t_pid, pid_data["omega_filtered"])
    ts_pid = min(m_pid["settling_time"], t_end - 0.3)
    ax1.annotate(
        f"PID  t_s ≈ {m_pid['settling_time']:.2f} s\n"
        f"      M_p ≈ {m_pid['overshoot_pct']:.1f} %",
        xy=(ts_pid, OMEGA_TARGET),
        xytext=(max(0.3, ts_pid - 1.5), 1.20*OMEGA_TARGET),
        arrowprops=dict(arrowstyle="->", color=COLORS["pid_filt"]),
        color=COLORS["pid_filt"], fontsize=9,
    )
    if ppo_data is not None:
        final_ppo = ppo_data["omega_filtered"][-1]
        m_ppo     = compute_metrics(ppo_data["time"], ppo_data["omega_filtered"])
        if final_ppo >= 0.90*OMEGA_TARGET:
            ts_ppo = min(m_ppo["settling_time"], t_end - 0.3)
            ax1.annotate(
                f"PPO t_s ≈ {m_ppo['settling_time']:.2f} s\n"
                f"      M_p ≈ {m_ppo['overshoot_pct']:.1f} %",
                xy=(ts_ppo, OMEGA_TARGET),
                xytext=(min(ts_ppo + 0.4, t_end - 0.9), 1.10*OMEGA_TARGET),
                arrowprops=dict(arrowstyle="->", color=COLORS["ppo_filt"]),
                color=COLORS["ppo_filt"], fontsize=9,
            )
        else:
            ax1.text(0.55, 0.30,
                     "⚠ Retrain with v4 motor_env.py\n  for smooth PPO voltage",
                     transform=ax1.transAxes, fontsize=9,
                     color=COLORS["ppo_filt"],
                     bbox=dict(boxstyle="round,pad=0.3", fc="white",
                               ec=COLORS["ppo_filt"], alpha=0.8))

    # Panel 2 — filtering
    ax2.plot(t_pid, pid_data["omega_noisy"],
             color=COLORS["noise"], lw=0.8, alpha=0.7, label="Noisy sensor (PID)")
    ax2.plot(t_pid, pid_data["omega_filtered"],
             color=COLORS["pid_filt"], label="Kalman output")
    ax2.plot(t_pid, pid_data["omega_true"],
             color=COLORS["pid_raw"], ls=":", lw=1.2, label="True ω (noiseless)")
    ax2.axhline(OMEGA_TARGET, color=COLORS["setpoint"], lw=1.2, ls="--")
    ax2.set_title("Sensor Noise & Kalman Filtering (Lissage / Échantillonnage)")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Angular velocity  ω [rad/s]")
    ax2.legend(loc="lower right", framealpha=0.9); ax2.set_xlim(0, t_end)

    axins = ax2.inset_axes([0.55, 0.08, 0.42, 0.42])
    ss = int(0.75 * len(t_pid))
    axins.plot(t_pid[ss:], pid_data["omega_noisy"][ss:],
               color=COLORS["noise"], lw=0.7, alpha=0.8)
    axins.plot(t_pid[ss:], pid_data["omega_filtered"][ss:],
               color=COLORS["pid_filt"])
    axins.axhline(OMEGA_TARGET, color=COLORS["setpoint"], lw=1.0, ls="--")
    axins.set_title("SS zoom", fontsize=8); axins.tick_params(labelsize=7)
    ax2.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.6)

    # Panel 3 — voltage
    ax3.plot(t_pid, pid_data["voltage"], color=COLORS["pid_filt"], label="PID  voltage")
    if ppo_data is not None:
        ax3.plot(ppo_data["time"], ppo_data["voltage"],
                 color=COLORS["ppo_filt"], label="PPO  voltage")
    ax3.axhline(V_MAX, color="k", lw=1.0, ls=":", alpha=0.6,
                label=f"V_max = {V_MAX} V")
    ax3.set_title("Control Effort — Variateur (Voltage Command)")
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Armature voltage  V [V]")
    ax3.legend(framealpha=0.9); ax3.set_xlim(0, t_end)
    ax3.set_ylim(-0.2, V_MAX * 1.12)

    # Panel 4 — current
    ax4.plot(t_pid, pid_data["current"], color=COLORS["pid_filt"], label="PID  current")
    if ppo_data is not None:
        ax4.plot(ppo_data["time"], ppo_data["current"],
                 color=COLORS["ppo_filt"], label="PPO  current")
    ax4.set_title("Armature Current  i(t)")
    ax4.set_xlabel("Time [s]"); ax4.set_ylabel("Current  i [A]")
    ax4.legend(framealpha=0.9); ax4.set_xlim(0, t_end)

    fig.legend(handles=[
        Line2D([0],[0], color=COLORS["pid_filt"], lw=2, label="PID Controller"),
        Line2D([0],[0], color=COLORS["ppo_filt"], lw=2, label="PPO RL Agent"),
        Line2D([0],[0], color=COLORS["setpoint"], lw=1.5, ls="--", label="Set-point"),
        Line2D([0],[0], color=COLORS["noise"],    lw=1, alpha=0.7, label="Noisy sensor"),
    ], loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005),
       framealpha=0.9, fontsize=10)

    fig.text(0.5, -0.01,
             "Parameters: J=0.01, b=0.1, K=0.01, R=1 Ω, L=0.5 H  |  "
             "dt=0.01 s (Euler)  |  σ_noise=0.02 rad/s  |  Target ω*=1.0 rad/s",
             ha="center", fontsize=8, color="grey", style="italic")

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n[OK] Figure saved → {save_path}")
    plt.show()


def plot_lissage(pid_data, save_path="lissage_comparison.png"):
    maf = MovingAverageFilter(window=15)
    kf2 = KalmanFilter1D(Q=1e-5, R_noise=4e-4)
    maf_trace = maf.filter_sequence(pid_data["omega_noisy"])
    kf2.reset()
    kf_trace  = kf2.filter_sequence(pid_data["omega_noisy"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pid_data["time"], pid_data["omega_noisy"],
            color=COLORS["noise"], lw=0.8, alpha=0.65, label="Raw noisy sensor")
    ax.plot(pid_data["time"], pid_data["omega_true"],
            "k:", lw=1.2, label="True ω (noiseless)")
    ax.plot(pid_data["time"], maf_trace,
            color=COLORS["ma_filter"], lw=1.8, label="Moving-Average filter (N=15)")
    ax.plot(pid_data["time"], kf_trace,
            color=COLORS["pid_filt"], lw=1.8, label="Kalman filter")
    ax.axhline(OMEGA_TARGET, color=COLORS["setpoint"], ls="--",
               lw=1.2, label="Set-point  ω* = 1.0 rad/s")
    ax.set_title(
        "Lissage / Échantillonnage — Kalman Filter vs. Moving-Average Filter\n"
        "(Applied to PID-controlled motor speed; σ_noise = 0.02 rad/s)", fontsize=12)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("ω [rad/s]")
    ax.legend(framealpha=0.9); ax.grid(True, alpha=0.35)
    ax.set_xlim(0, pid_data["time"][-1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Figure saved → {save_path}")
    plt.show()


# ===========================================================================
# Entry point
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DC Motor Control Suite — Evaluation & Plotting (v4)")
    print("=" * 60)

    print("\n[1/3] Running PID simulation …")
    pid_data = run_pid_simulation(Kp=5.0, Ki=12.0, Kd=0.80, use_kalman=True)
    print(f"      Final ω = {pid_data['omega_filtered'][-1]:.4f} rad/s")

    print("\n[2/3] Loading PPO model …")
    ppo_data = run_ppo_simulation(
        model_path   = "ppo_dc_motor_v4",
        vecnorm_path = "vec_normalize_v4.pkl",
        use_kalman   = True,
    )
    if ppo_data is not None:
        final = ppo_data["omega_filtered"][-1]
        v_std = float(np.std(np.diff(ppo_data["voltage"])))
        print(f"      Final ω = {final:.4f} rad/s")
        print(f"      Voltage Δ std = {v_std:.4f} V/step  "
              f"({'smooth ✓' if v_std < 0.5 else 'chattering ✗ — retrain with v4'})")

    print("\n[3/3] Performance metrics:")
    print_metrics_table(pid_data, ppo_data)

    plot_comparison(pid_data, ppo_data, save_path="control_comparison.png")

    print("\nGenerating Lissage figure …")
    plot_lissage(pid_data, save_path="lissage_comparison.png")
