#!/usr/bin/env python3
"""
Stewart Platform IMU System Identification Pipeline
Processes IMU data to estimate servo dynamics via system identification.

Pipeline: Raw IMU → Orientation (Kalman) → Servo Angles (IK) → System ID
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Tuple, Optional


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

@dataclass
class IMUConfig:
    """IMU sensor configuration and calibration parameters.

    Attributes:
        accel_scale: Accelerometer scale factor [m/s² per LSB]
        gyro_scale: Gyroscope scale factor [rad/s per LSB]
        accel_axis_flip: Axis orientation correction for accelerometer
        gyro_axis_flip: Axis orientation correction for gyroscope
        accel_rotation: Optional rotation matrix for accelerometer frame alignment
        gyro_rotation: Optional rotation matrix for gyroscope frame alignment
    """
    # LSM303DLHC Accelerometer: ±2g range, 12-bit resolution (after >>4 shift)
    # Sensitivity: 1 mg/LSB (datasheet verified)
    accel_scale: float = 0.001 * 9.81

    # L3GD20 Gyroscope: ±250 dps range
    # Sensitivity: 8.75 mdps/LSB (datasheet verified)
    gyro_scale: float = 0.00875 * np.pi / 180

    # Axis orientation: [X, Y, Z] multipliers (+1 or -1)
    # Default: Z-axis inverted based on physical mounting
    accel_axis_flip: np.ndarray = field(default_factory=lambda: np.array([1, 1, -1]))
    gyro_axis_flip: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1]))

    # Frame alignment: 3x3 rotation matrices for 90°/180° corrections
    # Set to None if no rotation needed
    accel_rotation: Optional[np.ndarray] = field(default_factory=lambda: np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ]))
    gyro_rotation: Optional[np.ndarray] = field(default_factory=lambda: np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ]))


@dataclass
class KalmanConfig:
    """Kalman filter noise covariance parameters.

    Attributes:
        process_noise_angle: Process noise for orientation states [rad²]
        process_noise_bias: Process noise for gyroscope bias states [(rad/s)²]
        accel_noise: Accelerometer measurement noise [m/s²]
        gyro_noise: Gyroscope measurement noise [rad/s]
    """
    process_noise_angle: float = 1e-5
    process_noise_bias: float = 1e-7
    accel_noise: float = 0.1
    gyro_noise: float = 0.01


@dataclass
class SystemIDConfig:
    """System identification configuration and transfer function parameters.

    Operating Modes:
        'fit':    Automatically estimate parameters via nonlinear least squares
        'manual': Use user-specified parameters (useful for validation/tuning)

    Transfer Function:
        H(s) = K·exp(-Td·s) / (τ·s + 1)

    Attributes:
        mode: Identification mode ('fit' or 'manual')
        K: Steady-state gain [dimensionless]
        tau: Time constant [s]
        delay: Time delay [s]
        initial_*: Initial guesses for curve fitting when mode='fit'
    """
    mode: str = 'manual'

    # Manual mode parameters
    K: float = 1.0
    tau: float = 0.1
    delay: float = 0.35 + 0.196 #0.159

    # Auto-fit mode initial guesses
    initial_K: float = 1.0
    initial_tau: float = 0.1
    initial_delay: float = 0.5


# ============================================================================
# IMU DATA PROCESSING
# ============================================================================

class IMUScaler:
    """Converts raw IMU sensor data to calibrated physical units."""

    def __init__(self, config: IMUConfig):
        self.config = config

    def scale_accel(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw accelerometer readings to m/s² with frame transformation.

        Args:
            raw: Raw accelerometer values [LSB]

        Returns:
            Calibrated acceleration vector [m/s²]
        """
        scaled = raw * self.config.accel_scale
        scaled = scaled * self.config.accel_axis_flip

        if self.config.accel_rotation is not None:
            scaled = scaled @ self.config.accel_rotation.T

        return scaled

    def scale_gyro(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw gyroscope readings to rad/s with frame transformation.

        Args:
            raw: Raw gyroscope values [LSB]

        Returns:
            Calibrated angular velocity vector [rad/s]
        """
        scaled = raw * self.config.gyro_scale
        scaled = scaled * self.config.gyro_axis_flip

        if self.config.gyro_rotation is not None:
            scaled = scaled @ self.config.gyro_rotation.T

        return scaled


# ============================================================================
# ORIENTATION ESTIMATION
# ============================================================================

class OrientationKalmanFilter:
    """Extended Kalman Filter for roll and pitch estimation from IMU measurements.

    State vector: [roll, pitch, gyro_bias_x, gyro_bias_y]

    Features:
        - Automatic gravity vector zeroing at initialization
        - Gyroscope bias estimation
        - Complementary sensor fusion (gyro prediction, accel correction)
    """

    def __init__(self, config: KalmanConfig):
        self.config = config

        # State: [roll, pitch, gyro_bias_x, gyro_bias_y]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 0.1

        # Process noise covariance
        self.Q = np.diag([
            config.process_noise_angle,
            config.process_noise_angle,
            config.process_noise_bias,
            config.process_noise_bias
        ])

        # Measurement noise covariance
        self.R = np.diag([
            config.accel_noise ** 2,
            config.accel_noise ** 2
        ])

        self.initialized = False
        self.initial_accel = None

    def initialize(self, accel: np.ndarray):
        """Initialize filter state from first accelerometer reading.

        Args:
            accel: Initial acceleration measurement [m/s²]
        """
        if not self.initialized:
            ax, ay, az = accel
            roll0 = np.arctan2(ay, az)
            pitch0 = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

            self.state[0] = roll0
            self.state[1] = pitch0
            self.initial_accel = accel.copy()
            self.initialized = True

    def predict(self, gyro: np.ndarray, dt: float):
        """Prediction step using gyroscope measurements.

        Args:
            gyro: Angular velocity measurement [rad/s]
            dt: Time step [s]
        """
        gx, gy = gyro[0], gyro[1]

        # Bias-corrected angular velocity
        gx_corrected = gx - self.state[2]
        gy_corrected = gy - self.state[3]

        # State propagation
        self.state[0] += gx_corrected * dt
        self.state[1] += gy_corrected * dt

        # Jacobian of state transition
        F = np.array([
            [1, 0, -dt, 0],
            [0, 1, 0, -dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update(self, accel: np.ndarray):
        """Update step using accelerometer measurements.

        Args:
            accel: Acceleration measurement [m/s²]
        """
        ax, ay, az = accel

        # Tilt angles from accelerometer
        roll_meas = np.arctan2(ay, az)
        pitch_meas = np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2))

        # Remove initial gravity offset
        if self.initial_accel is not None:
            roll_init = np.arctan2(self.initial_accel[1], self.initial_accel[2])
            pitch_init = np.arctan2(-self.initial_accel[0],
                                    np.sqrt(self.initial_accel[1] ** 2 + self.initial_accel[2] ** 2))
            roll_meas -= roll_init
            pitch_meas -= pitch_init

        z = np.array([roll_meas, pitch_meas])

        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Innovation
        y = z - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State and covariance update
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_orientation(self) -> Tuple[float, float]:
        """Return current orientation estimate.

        Returns:
            Tuple of (roll, pitch) in radians
        """
        return self.state[0], self.state[1]


# ============================================================================
# INVERSE KINEMATICS
# ============================================================================

class StewartPlatformIK:
    """Inverse kinematics solver for Stewart platform using Eisele's method.

    Computes servo angles required to achieve desired platform pose.

    Args:
        horn_length: Servo horn length [mm]
        rod_length: Connecting rod length [mm]
        base: Base platform radius [mm]
        base_anchors: Base anchor offset [mm]
        platform: Moving platform radius [mm]
        platform_anchors: Platform anchor offset [mm]
        top_surface_offset: Platform surface to anchor center offset [mm]
    """

    def __init__(self, horn_length=31.75, rod_length=145.0, base=73.025,
                 base_anchors=36.8893, platform=67.775, platform_anchors=12.7,
                 top_surface_offset=26.0):
        self.horn_length = horn_length
        self.rod_length = rod_length
        self.base = base
        self.base_anchors = base_anchors
        self.platform = platform
        self.platform_anchors = platform_anchors
        self.top_surface_offset = top_surface_offset

        base_angels = np.array([-np.pi / 2, np.pi / 6, np.pi * 5 / 6])
        platform_angels = np.array([-np.pi * 5 / 6, -np.pi / 6, np.pi / 2])

        self.base_anchors = self._calculate_home_coordinates(self.base, self.base_anchors, base_angels)
        platform_anchors_out = self._calculate_home_coordinates(self.platform, self.platform_anchors,
                                                                platform_angels)
        self.platform_anchors = np.roll(platform_anchors_out, shift=-1, axis=0)
        self.beta_angles = self._calculate_beta_angles()

        base_pos = self.base_anchors[0]
        platform_pos = self.platform_anchors[0]

        horn_end_x = base_pos[0] + self.horn_length * np.cos(self.beta_angles[0])
        horn_end_y = base_pos[1] + self.horn_length * np.sin(self.beta_angles[0])

        dx = platform_pos[0] - horn_end_x
        dy = platform_pos[1] - horn_end_y
        horiz_dist_sq = dx ** 2 + dy ** 2

        self.home_height = np.sqrt(self.rod_length ** 2 - horiz_dist_sq)
        self.home_height_top_surface = self.home_height + self.top_surface_offset

    def _calculate_home_coordinates(self, l, d, phi):
        angels = np.array([-np.pi / 2, np.pi / 2])
        xy = np.zeros((6, 3))
        for i in range(len(phi)):
            for j in range(len(angels)):
                x = l * np.cos(phi[i]) + d * np.cos(phi[i] + angels[j])
                y = l * np.sin(phi[i]) + d * np.sin(phi[i] + angels[j])
                xy[i * 2 + j] = np.array([x, y, 0])
        return xy

    def _calculate_beta_angles(self):
        beta_angles = np.zeros(6)
        beta_angles[0] = 0
        beta_angles[1] = np.pi

        dx_23 = self.base_anchors[3, 0] - self.base_anchors[2, 0]
        dy_23 = self.base_anchors[3, 1] - self.base_anchors[2, 1]
        angle_23 = np.arctan2(dy_23, dx_23)
        beta_angles[2] = angle_23
        beta_angles[3] = angle_23 + np.pi

        dx_54 = self.base_anchors[4, 0] - self.base_anchors[5, 0]
        dy_54 = self.base_anchors[4, 1] - self.base_anchors[5, 1]
        angle_54 = np.arctan2(dy_54, dx_54)
        beta_angles[5] = angle_54
        beta_angles[4] = angle_54 + np.pi

        return beta_angles

    def calculate_servo_angles(self, translation: np.ndarray, rotation: np.ndarray,
                               use_top_surface_offset: bool = True):
        quat = self._euler_to_quaternion(rotation)

        if use_top_surface_offset:
            offset_platform_frame = np.array([0, 0, -self.top_surface_offset])
            offset_world_frame = self._rotate_vector(offset_platform_frame, quat)
            anchor_center_translation = translation + offset_world_frame
        else:
            anchor_center_translation = translation

        angles = np.zeros(6)

        for k in range(6):
            p_world = anchor_center_translation + self._rotate_vector(self.platform_anchors[k], quat)
            leg = p_world - self.base_anchors[k]
            leg_length_sq = np.dot(leg, leg)

            e_k = 2 * self.horn_length * leg[2]
            f_k = 2 * self.horn_length * (
                    np.cos(self.beta_angles[k]) * leg[0] +
                    np.sin(self.beta_angles[k]) * leg[1]
            )
            g_k = leg_length_sq - (self.rod_length ** 2 - self.horn_length ** 2)

            sqrt_term = e_k ** 2 + f_k ** 2
            if sqrt_term < 1e-6:
                return None

            ratio = g_k / np.sqrt(sqrt_term)
            if abs(ratio) > 1.0:
                return None

            alpha_k = np.arcsin(ratio) - np.arctan2(f_k, e_k)
            angles[k] = np.degrees(alpha_k)

            if abs(angles[k]) > 40:
                return None

        return -angles

    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        rx, ry, rz = euler
        cy = np.cos(rz * 0.5)
        sy = np.sin(rz * 0.5)
        cp = np.cos(ry * 0.5)
        sp = np.sin(ry * 0.5)
        cr = np.cos(rx * 0.5)
        sr = np.sin(rx * 0.5)

        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])

    def _rotate_vector(self, v: np.ndarray, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        vx, vy, vz = v

        return np.array([
            vx * (w * w + x * x - y * y - z * z) + vy * (2 * x * y - 2 * w * z) + vz * (2 * x * z + 2 * w * y),
            vx * (2 * x * y + 2 * w * z) + vy * (w * w - x * x + y * y - z * z) + vz * (2 * y * z - 2 * w * x),
            vx * (2 * x * z - 2 * w * y) + vy * (2 * y * z + 2 * w * x) + vz * (w * w - x * x - y * y + z * z)
        ])


# ============================================================================
# SYSTEM IDENTIFICATION
# ============================================================================

# ============================================================================
# SYSTEM IDENTIFICATION
# ============================================================================

class FirstOrderSystemID:
    """First-order transfer function identification.

    Model: H(s) = K·exp(-Td·s) / (τ·s + 1)

    Parameters:
        K: Steady-state gain [dimensionless]
        τ: Time constant [s]
        Td: Time delay [s]
    """

    def __init__(self, config: SystemIDConfig):
        self.config = config
        self.params = None

        if config.mode == 'manual':
            self.params = np.array([config.K, config.tau, config.delay])

    @staticmethod
    def first_order_response(t, K, tau, delay):
        """Compute first-order step response.

        Args:
            t: Time vector [s]
            K: Steady-state gain
            tau: Time constant [s]
            delay: Time delay [s]

        Returns:
            Step response amplitude
        """
        response = np.zeros_like(t)
        mask = t >= delay
        response[mask] = K * (1 - np.exp(-(t[mask] - delay) / tau))
        return response

    def fit(self, time: np.ndarray, commanded: np.ndarray, actual: np.ndarray) -> dict:
        """Identify transfer function parameters from step response data.

        Args:
            time: Time vector [s]
            commanded: Commanded input signal
            actual: Measured output signal

        Returns:
            Dictionary containing identified parameters and fit quality (R²)
        """
        if self.config.mode == 'manual':
            K, tau, delay = self.params

            step_magnitude = np.abs(commanded[-1] - commanded[0])
            actual_normalized = (actual - actual[0]) / step_magnitude
            time_normalized = time - time[0]

            predicted = self.first_order_response(time_normalized, K, tau, delay)
            residuals = actual_normalized - predicted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((actual_normalized - np.mean(actual_normalized)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'K': K,
                'tau': tau,
                'delay': delay,
                'r_squared': r_squared,
                'params': self.params,
                'mode': 'manual'
            }

        step_magnitude = np.abs(commanded[-1] - commanded[0])

        if step_magnitude < 0.1:
            return None

        actual_normalized = (actual - actual[0]) / step_magnitude
        time_normalized = time - time[0]

        try:
            popt, pcov = curve_fit(
                self.first_order_response,
                time_normalized,
                actual_normalized,
                p0=[self.config.initial_K, self.config.initial_tau, self.config.initial_delay],
                bounds=([0.5, 0.001, 0], [1.5, 1.0, 0.5]),
                maxfev=10000
            )

            K, tau, delay = popt

            residuals = actual_normalized - self.first_order_response(time_normalized, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((actual_normalized - np.mean(actual_normalized)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            self.params = popt

            return {
                'K': K,
                'tau': tau,
                'delay': delay,
                'r_squared': r_squared,
                'params': popt,
                'mode': 'fit'
            }
        except Exception as e:
            print(f"Parameter fitting failed: {e}")
            return None

    def predict(self, time: np.ndarray, commanded: np.ndarray) -> np.ndarray:
        """Generate model prediction for given input signal.

        Args:
            time: Time vector [s]
            commanded: Commanded input signal

        Returns:
            Predicted output signal
        """
        if self.params is None:
            return None

        K, tau, delay = self.params
        step_magnitude = commanded[-1] - commanded[0]
        time_normalized = time - time[0]

        response = self.first_order_response(time_normalized, K, tau, delay)
        return commanded[0] + response * step_magnitude


class SecondOrderSystemID:
    """Second-order transfer function identification with time delay.

    Model: H(s) = K·ωn²·exp(-Td·s) / (s² + 2ζωn·s + ωn²)

    Parameters:
        K: Steady-state gain [dimensionless]
        ζ: Damping ratio [dimensionless]
        ωn: Natural frequency [rad/s]
        Td: Time delay [s]
    """

    def __init__(self, config: SystemIDConfig):
        self.config = config
        self.params = None

        if config.mode == 'manual':
            self.params = np.array([config.K, config.zeta, config.wn, config.delay])

    @staticmethod
    def second_order_response(t, K, zeta, wn, delay):
        """Compute second-order step response.

        Args:
            t: Time vector [s]
            K: Steady-state gain
            zeta: Damping ratio
            wn: Natural frequency [rad/s]
            delay: Time delay [s]

        Returns:
            Step response amplitude
        """
        response = np.zeros_like(t)
        mask = t >= delay
        t_delayed = t[mask] - delay

        if zeta < 1:
            wd = wn * np.sqrt(1 - zeta ** 2)
            response[mask] = K * (1 - np.exp(-zeta * wn * t_delayed) *
                                  (np.cos(wd * t_delayed) +
                                   (zeta / np.sqrt(1 - zeta ** 2)) * np.sin(wd * t_delayed)))
        elif zeta == 1:
            response[mask] = K * (1 - np.exp(-wn * t_delayed) * (1 + wn * t_delayed))
        else:
            s1 = -zeta * wn + wn * np.sqrt(zeta ** 2 - 1)
            s2 = -zeta * wn - wn * np.sqrt(zeta ** 2 - 1)
            response[mask] = K * (1 - (s1 * np.exp(s2 * t_delayed) - s2 * np.exp(s1 * t_delayed)) / (s1 - s2))

        return response

    def fit(self, time: np.ndarray, commanded: np.ndarray, actual: np.ndarray) -> dict:
        """Identify transfer function parameters from step response data.

        Args:
            time: Time vector [s]
            commanded: Commanded input signal
            actual: Measured output signal

        Returns:
            Dictionary containing identified parameters and fit quality (R²)
        """
        if self.config.mode == 'manual':
            K, zeta, wn, delay = self.params

            step_magnitude = np.abs(commanded[-1] - commanded[0])
            actual_normalized = (actual - actual[0]) / step_magnitude
            time_normalized = time - time[0]

            predicted = self.second_order_response(time_normalized, K, zeta, wn, delay)
            residuals = actual_normalized - predicted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((actual_normalized - np.mean(actual_normalized)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                'K': K,
                'zeta': zeta,
                'wn': wn,
                'delay': delay,
                'r_squared': r_squared,
                'params': self.params,
                'mode': 'manual'
            }

        step_magnitude = np.abs(commanded[-1] - commanded[0])

        if step_magnitude < 0.1:
            return None

        actual_normalized = (actual - actual[0]) / step_magnitude
        time_normalized = time - time[0]

        try:
            popt, pcov = curve_fit(
                self.second_order_response,
                time_normalized,
                actual_normalized,
                p0=[self.config.initial_K, self.config.initial_zeta, self.config.initial_wn, self.config.initial_delay],
                bounds=([0.5, 0.1, 1.0, 0], [1.5, 2.0, 50.0, 0.5]),
                maxfev=10000
            )

            K, zeta, wn, delay = popt

            residuals = actual_normalized - self.second_order_response(time_normalized, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((actual_normalized - np.mean(actual_normalized)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            self.params = popt

            return {
                'K': K,
                'zeta': zeta,
                'wn': wn,
                'delay': delay,
                'r_squared': r_squared,
                'params': popt,
                'mode': 'fit'
            }
        except Exception as e:
            print(f"Parameter fitting failed: {e}")
            return None

    def predict(self, time: np.ndarray, commanded: np.ndarray) -> np.ndarray:
        """Generate model prediction for given input signal.

        Args:
            time: Time vector [s]
            commanded: Commanded input signal

        Returns:
            Predicted output signal
        """
        if self.params is None:
            return None

        K, zeta, wn, delay = self.params
        step_magnitude = commanded[-1] - commanded[0]
        time_normalized = time - time[0]

        response = self.second_order_response(time_normalized, K, zeta, wn, delay)
        return commanded[0] + response * step_magnitude


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_experiment(csv_file: str):
    """Execute complete system identification pipeline on experimental data.

    Args:
        csv_file: Path to CSV file containing experiment data

    Pipeline stages:
        1. Load and validate CSV data
        2. Scale and transform raw IMU measurements
        3. Estimate orientation using Kalman filter
        4. Compute servo angles via inverse kinematics
        5. Identify transfer function parameters
        6. Generate comparative plots
    """

    print(f"Processing: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"  Loaded {len(df)} samples")

    imu_config = IMUConfig()
    kalman_config = KalmanConfig()
    sysid_config = SystemIDConfig()

    scaler = IMUScaler(imu_config)
    kalman = OrientationKalmanFilter(kalman_config)
    ik = StewartPlatformIK()
    sysid = FirstOrderSystemID(sysid_config)

    print(f"  System identification mode: {sysid_config.mode}")

    time = df['timestamp_pc'].values - df['timestamp_pc'].values[0]

    rx_cmd = np.radians(df['rx'].values)
    ry_cmd = np.radians(df['ry'].values)

    accel_raw = df[['accel_x', 'accel_y', 'accel_z']].values
    gyro_raw = df[['gyro_x', 'gyro_y', 'gyro_z']].values

    theta_cmd = df[['theta0', 'theta1', 'theta2', 'theta3', 'theta4', 'theta5']].values

    print("  Processing IMU data...")
    rx_est = np.zeros(len(df))
    ry_est = np.zeros(len(df))

    for i in range(len(df)):
        accel = scaler.scale_accel(accel_raw[i])
        gyro = scaler.scale_gyro(gyro_raw[i])

        if i == 0:
            kalman.initialize(accel)
            dt = 0.001
        else:
            dt = time[i] - time[i - 1]

        kalman.predict(gyro, dt)
        kalman.update(accel)

        rx_est[i], ry_est[i] = kalman.get_orientation()

    print("  Computing inverse kinematics...")
    theta_est = np.zeros((len(df), 6))

    home_z = ik.home_height_top_surface

    for i in range(len(df)):
        translation = np.array([0, 0, home_z])
        rotation = np.array([rx_est[i], ry_est[i], 0])

        angles = ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

        if angles is not None:
            theta_est[i] = angles
        else:
            theta_est[i] = theta_est[i - 1] if i > 0 else np.zeros(6)

    print("  Performing system identification...")

    rx_diff = np.diff(rx_cmd)
    step_idx = np.where(np.abs(rx_diff) > np.radians(1))[0]

    fit_result = None
    rx_model = None
    ry_model = None
    theta_model = None

    if len(step_idx) > 0:
        step_start = max(0, step_idx[0] - 10)
        step_end = min(len(time), step_idx[0] + int(2.0 / np.mean(np.diff(time))))

        theta_cmd_avg = np.mean(theta_cmd[step_start:step_end], axis=1)
        theta_est_avg = np.mean(theta_est[step_start:step_end], axis=1)

        fit_result = sysid.fit(
            time[step_start:step_end],
            theta_cmd_avg,
            theta_est_avg
        )

        if fit_result:
            mode_str = fit_result.get('mode', 'fit')

            print(f"\n  System ID Results (All Servos) - Mode: {mode_str.upper()}")
            print(f"    K (Gain):       {fit_result['K']:.4f}")
            print(f"    τ (Time const): {fit_result['tau']:.4f} s")
            print(f"    Td (Delay):     {fit_result['delay']:.4f} s")
            print(f"    R²:             {fit_result['r_squared']:.4f}")

            if mode_str == 'manual':
                print(f"    Note: Using manually specified parameters")
            else:
                print(f"    Note: Parameters auto-fitted from step response data")

            print("  Generating model predictions...")

            rx_model = sysid.predict(time, rx_cmd)
            ry_model = sysid.predict(time, ry_cmd)

            theta_model = np.zeros_like(theta_cmd)
            for i in range(6):
                pred = sysid.predict(time, theta_cmd[:, i])
                if pred is not None:
                    theta_model[:, i] = pred

    print("  Generating plots...")

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(time, np.degrees(rx_cmd), 'b-', label='Commanded', linewidth=2)
    ax1.plot(time, np.degrees(rx_est), 'r--', label='IMU Estimated', linewidth=1.5)
    if rx_model is not None:
        ax1.plot(time, np.degrees(rx_model), 'g:', label='Model Prediction', linewidth=2)
    ax1.set_ylabel('Roll (rx) [deg]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Orientation Comparison: Commanded vs IMU vs Model')

    ax2.plot(time, np.degrees(ry_cmd), 'b-', label='Commanded', linewidth=2)
    ax2.plot(time, np.degrees(ry_est), 'r--', label='IMU Estimated', linewidth=1.5)
    if ry_model is not None:
        ax2.plot(time, np.degrees(ry_model), 'g:', label='Model Prediction', linewidth=2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Pitch (ry) [deg]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    fig2, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(6):
        axes[i].plot(time, theta_cmd[:, i], 'b-', label='Commanded', linewidth=2)
        axes[i].plot(time, theta_est[:, i], 'r--', label='IMU-IK', linewidth=1.5)
        if theta_model is not None:
            axes[i].plot(time, theta_model[:, i], 'g:', label='Model', linewidth=2)
        axes[i].set_ylabel(f'θ{i} [deg]')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)

        if i >= 4:
            axes[i].set_xlabel('Time [s]')

    fig2.suptitle('Servo Angles: Commanded vs IMU-IK vs Model', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Stewart Platform System Identification Tool")
        print("=" * 70)
        print("\nUsage:")
        print("  python script.py <experiment_csv_file>")
        print("\nConfiguration:")
        print("  Edit dataclass definitions at top of script:")
        print("\n  1. IMUConfig")
        print("     - Sensor scaling factors (LSM303DLHC + L3GD20)")
        print("     - Axis orientation corrections")
        print("     - Frame rotation matrices")
        print("\n  2. KalmanConfig")
        print("     - Process noise covariance")
        print("     - Measurement noise covariance")
        print("\n  3. SystemIDConfig")
        print("     - mode: 'fit' (auto) or 'manual' (user-specified)")
        print("     - Parameters: K, τ, Td")
        print("\nTransfer Function:")
        print("  H(s) = K·exp(-Td·s) / (τ·s + 1)")
        print("\nOutput:")
        print("  - Console: Identified parameters and fit quality (R²)")
        print("  - Plots: Orientation and servo angle comparisons")
        print("=" * 70)
        sys.exit(1)

    csv_file = sys.argv[1]
    process_experiment(csv_file)