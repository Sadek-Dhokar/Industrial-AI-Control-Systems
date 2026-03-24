"""
signal_processing.py
====================
Signal Smoothing / Lissage Module for DC Motor Speed Control
=============================================================

This module implements two complementary digital filter designs for
cleaning the noisy tachometer signal produced by the DCMotorEnv sensor
model.  It fulfils the "Lissage / Échantillonnage" requirement of the
project specification.

Filter 1 — Moving-Average (MA) Filter
--------------------------------------
    y[k] = (1/N) · Σ_{j=0}^{N-1}  x[k-j]
Simple, zero-tuning-parameter smoother.  Introduces a group delay of
(N-1)/2 samples.  Suitable for offline analysis and as a baseline
comparator for the Kalman filter.

Filter 2 — Discrete Kalman Filter (1-D, Constant-Velocity Model)
-----------------------------------------------------------------
Optimal linear estimator under Gaussian noise assumptions.

State model:
    x[k] = A · x[k-1] + w[k],    w ~ N(0, Q)   (process noise)
    z[k] = H · x[k]   + v[k],    v ~ N(0, R_n)  (measurement noise)

For a nearly-constant speed assumption we use:
    A = 1 (identity), H = 1

The scalar Riccati recursion is:
    Predict:   P⁻ = P + Q
    Update:    K_g = P⁻ / (P⁻ + R_n)
               x̂   = x̂ + K_g · (z - x̂)
               P   = (1 - K_g) · P⁻

Tuning:
    Q (process noise variance) – how rapidly the true speed can change
    R_n (sensor noise variance) – matches the encoder noise power
    Large Q/R_n → filter tracks fast, less smoothing
    Small Q/R_n → filter lags more, heavier smoothing

Usage
-----
    from signal_processing import KalmanFilter1D, MovingAverageFilter

    kf  = KalmanFilter1D(Q=1e-4, R_noise=4e-4)
    maf = MovingAverageFilter(window=10)

    for noisy_measurement in stream:
        filtered_kf  = kf.update(noisy_measurement)
        filtered_maf = maf.update(noisy_measurement)
"""

from collections import deque
import numpy as np


# ===========================================================================
class MovingAverageFilter:
    """
    Causal finite-impulse-response (FIR) moving-average filter.

    Parameters
    ----------
    window : int  – number of past samples to average (odd preferred
                    so that group delay is a half-integer of samples)
    """

    def __init__(self, window: int = 10):
        if window < 1:
            raise ValueError("window must be ≥ 1")
        self.window = window
        # deque with fixed max-length acts as an efficient circular buffer
        self._buffer = deque(maxlen=window)

    # ------------------------------------------------------------------
    def update(self, measurement: float) -> float:
        """
        Accept one new sample and return the filtered estimate.

        Parameters
        ----------
        measurement : float  – raw (noisy) speed sample [rad/s]

        Returns
        -------
        float – smoothed speed estimate [rad/s]
        """
        self._buffer.append(measurement)
        return float(np.mean(self._buffer))

    # ------------------------------------------------------------------
    def reset(self):
        """Clear the internal buffer (call between episodes)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    def filter_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Convenience method: filter an entire array in one call.
        Resets the buffer before processing.

        Parameters
        ----------
        sequence : np.ndarray  – 1-D array of raw samples

        Returns
        -------
        np.ndarray – filtered array of same length
        """
        self.reset()
        return np.array([self.update(x) for x in sequence])


# ===========================================================================
class KalmanFilter1D:
    """
    Scalar (1-D state, 1-D observation) discrete-time Kalman filter.

    Assumes a constant-velocity (random-walk) state model:
        x[k+1] = x[k] + w[k]    (motor speed changes slowly)
        z[k]   = x[k] + v[k]    (tachometer measurement)

    Parameters
    ----------
    Q       : float  – process noise variance  (≈ variance of Δω per step)
    R_noise : float  – measurement noise variance (= σ_noise²)
    x0      : float  – initial state estimate [rad/s]
    P0      : float  – initial error covariance (uncertainty in x0)
    """

    def __init__(self,
                 Q:       float = 1e-4,
                 R_noise: float = 4e-4,
                 x0:      float = 0.0,
                 P0:      float = 1.0):
        self.Q       = Q
        self.R_noise = R_noise

        # State estimate and error covariance
        self.x_est = float(x0)
        self.P_est = float(P0)

        # Diagnostics: Kalman gain history for plotting
        self._gain_history = []

    # ------------------------------------------------------------------
    def update(self, measurement: float) -> float:
        """
        Perform one predict–correct cycle of the Kalman filter.

        Predict step
        ~~~~~~~~~~~~
            x⁻[k] = A · x̂[k-1]     (A=1: speed assumed constant)
            P⁻[k] = P[k-1] + Q

        Correct step
        ~~~~~~~~~~~~
            K_g      = P⁻[k] / (P⁻[k] + R_n)
            x̂[k]     = x⁻[k] + K_g · (z[k] - x⁻[k])
            P[k]     = (1 - K_g) · P⁻[k]

        Parameters
        ----------
        measurement : float  – noisy sensor reading z[k] [rad/s]

        Returns
        -------
        float – posterior estimate x̂[k] [rad/s]
        """
        # ---- Predict --------------------------------------------------
        x_pred = self.x_est                   # A = 1
        P_pred = self.P_est + self.Q          # propagate uncertainty

        # ---- Update (correct) -----------------------------------------
        innovation = measurement - x_pred    # ỹ = z - H·x⁻
        S          = P_pred + self.R_noise   # innovation covariance
        K_gain     = P_pred / S              # Kalman gain  K = P⁻H'S⁻¹

        self.x_est = x_pred + K_gain * innovation
        self.P_est = (1.0 - K_gain) * P_pred   # Joseph form for scalar case

        # Guard against numerical drift
        self.x_est = max(self.x_est, 0.0)      # speed ≥ 0

        self._gain_history.append(K_gain)
        return float(self.x_est)

    # ------------------------------------------------------------------
    def reset(self, x0: float = 0.0, P0: float = 1.0):
        """Re-initialise the filter state (call between episodes)."""
        self.x_est = float(x0)
        self.P_est = float(P0)
        self._gain_history.clear()

    # ------------------------------------------------------------------
    def filter_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Batch-process an entire 1-D sequence.
        Resets the filter before processing.

        Parameters
        ----------
        sequence : np.ndarray  – raw noisy speed measurements

        Returns
        -------
        np.ndarray – Kalman-filtered estimates (same length)
        """
        self.reset()
        return np.array([self.update(z) for z in sequence])

    # ------------------------------------------------------------------
    @property
    def gain_history(self) -> np.ndarray:
        """Return the array of Kalman gains (useful for diagnostics)."""
        return np.array(self._gain_history)


# ===========================================================================
# Quick smoke-test
# ===========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng  = np.random.default_rng(42)
    t    = np.linspace(0, 5, 500)
    # Simulate a first-order step response (true signal)
    tau  = 1.0
    true_signal = 1.0 * (1.0 - np.exp(-t / tau))
    noisy        = true_signal + rng.normal(0, 0.02, len(t))

    kf  = KalmanFilter1D(Q=1e-5, R_noise=4e-4)
    maf = MovingAverageFilter(window=15)

    kf_out  = kf.filter_sequence(noisy)
    maf_out = maf.filter_sequence(noisy)

    plt.figure(figsize=(10, 4))
    plt.plot(t, noisy,      alpha=0.4, label="Noisy (raw)")
    plt.plot(t, true_signal, "k--",   label="True signal")
    plt.plot(t, maf_out,    label=f"MA (N={maf.window})")
    plt.plot(t, kf_out,     label="Kalman Filter")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [rad/s]")
    plt.title("Filter Comparison — signal_processing.py smoke test")
    plt.legend()
    plt.tight_layout()
    plt.savefig("filter_smoketest.png", dpi=150)
    plt.show()
    print("Smoke test passed.  Figure saved to filter_smoketest.png")
