#!/usr/bin/env python3
"""
Shared utilities and constants for Stewart Platform simulators.

Centralized configuration for all system parameters:
- Physical constants and limits
- Controller defaults (PID, LQR)
- Kalman filter settings
- Camera model parameters
- Platform geometry
- Simulation settings
- Performance optimization
"""

import numpy as np

# ============================================================================
# PLATFORM VERSION SELECTION
# ============================================================================
PLATFORM_VERSION = 'V1'  # Options: 'V1' (200x200mm square) or 'V2' (larger platform)

# ============================================================================
# PHYSICAL CONSTANTS AND LIMITS
# ============================================================================

MAX_TILT_ANGLE_DEG = 15.0  # Maximum platform tilt magnitude (degrees)
MAX_SERVO_ANGLE_DEG = 70.0  # Maximum individual servo angle (degrees)

# Platform boundaries (version-dependent)
if PLATFORM_VERSION == 'V1':
    PLATFORM_HALF_SIZE_MM = 100.0  # V1: 200x200mm square platform (half = 100mm)
    PLATFORM_RADIUS_MM = 100.0  # V1: Square platform uses half-size for radius (boundaries are square)
else:  # V2
    PLATFORM_HALF_SIZE_MM = 150.0  # V2: Larger platform
    PLATFORM_RADIUS_MM = 150.0  # V2: Circular platform radius

# ============================================================================
# STEWART PLATFORM GEOMETRY CONFIGURATION
# ============================================================================

class StewartPlatformConfig:
    """Stewart platform geometric parameters (measured from actual hardware)."""

    # Platform version-specific parameters
    if PLATFORM_VERSION == 'V1':
        # V1 Platform (original 200x200mm square platform)
        HORN_LENGTH_MM = 31.75  # Servo horn length (mm)
        ROD_LENGTH_MM = 145.0  # Push rod length (mm)
        BASE_RADIUS_MM = 73.025  # Base circle radius (mm)
        BASE_ANCHORS_OFFSET_MM = 36.8893  # Anchor offset from base circle (mm)
        PLATFORM_RADIUS_MM = 67.775  # Platform circle radius (mm)
        PLATFORM_ANCHORS_OFFSET_MM = 12.7  # Anchor offset from platform circle (mm)
        TOP_SURFACE_OFFSET_MM = 26.0  # Distance from platform anchors to top surface (mm)
    else:  # V2
        # V2 Platform (current larger platform)
        HORN_LENGTH_MM = 45.3722  # Servo horn length (mm) - measured from hardware
        ROD_LENGTH_MM = 205.0  # Push rod length (mm) - measured from hardware
        BASE_RADIUS_MM = 116.3525  # Base circle radius (mm) - 86.6025 + 18.75 + 11
        BASE_ANCHORS_OFFSET_MM = 64.75  # Anchor offset from base circle (mm) - measured from hardware
        PLATFORM_RADIUS_MM = 84.0759  # Platform circle radius (mm) - measured from hardware
        PLATFORM_ANCHORS_OFFSET_MM = 12.5  # Anchor offset from platform circle (mm) - measured from hardware
        TOP_SURFACE_OFFSET_MM = 38.0  # Distance from platform anchors to top surface (mm) - measured from hardware

    @classmethod
    def as_dict(cls):
        """Return platform parameters as dictionary for StewartPlatformIK."""
        return {
            'horn_length': cls.HORN_LENGTH_MM,
            'rod_length': cls.ROD_LENGTH_MM,
            'base': cls.BASE_RADIUS_MM,
            'base_anchors': cls.BASE_ANCHORS_OFFSET_MM,
            'platform': cls.PLATFORM_RADIUS_MM,
            'platform_anchors': cls.PLATFORM_ANCHORS_OFFSET_MM,
            'top_surface_offset': cls.TOP_SURFACE_OFFSET_MM
        }

# ============================================================================
# COLOR SCHEME CONFIGURATION
# ============================================================================

class ColorScheme:
    """Dark theme color scheme for GUI."""

    BG = '#1e1e1e'
    PANEL_BG = '#2d2d2d'
    WIDGET_BG = '#3d3d3d'
    FG = '#e0e0e0'
    HIGHLIGHT = '#007acc'
    BUTTON_BG = '#0e639c'
    BUTTON_FG = '#ffffff'
    ENTRY_BG = '#3d3d3d'
    BORDER = '#555555'
    SUCCESS = '#4ec9b0'
    WARNING = '#ce9178'

    @classmethod
    def as_dict(cls):
        """Return colors as dictionary."""
        return {
            'bg': cls.BG,
            'panel_bg': cls.PANEL_BG,
            'widget_bg': cls.WIDGET_BG,
            'fg': cls.FG,
            'highlight': cls.HIGHLIGHT,
            'button_bg': cls.BUTTON_BG,
            'button_fg': cls.BUTTON_FG,
            'entry_bg': cls.ENTRY_BG,
            'border': cls.BORDER,
            'success': cls.SUCCESS,
            'warning': cls.WARNING
        }

# ============================================================================
# CONTROL LOOP CONFIGURATION
# ============================================================================

class ControlLoopConfig:
    """Configuration for real-time control loop (hardware mode)."""
    DEFAULT_FREQUENCY_HZ = 250  # Default control loop frequency
    MIN_FREQUENCY_HZ = 50  # Minimum control loop frequency
    MAX_FREQUENCY_HZ = 500  # Maximum control loop frequency
    FREQUENCY_HZ = DEFAULT_FREQUENCY_HZ  # Current control loop frequency (configurable)
    INTERVAL_S = 1.0 / FREQUENCY_HZ  # Control loop period
    IK_TIMEOUT_S = 0.004  # Maximum time allowed for IK calculation (4ms for 500Hz)
    MAX_LOOP_TIME_S = 0.002  # Maximum acceptable loop time (2ms for 500Hz)


class GUIConfig:
    """Configuration for GUI updates (applies to both sim and hardware)."""
    UPDATE_HZ = 5  # GUI refresh rate
    UPDATE_INTERVAL_MS = 200  # GUI update interval (milliseconds)
    LOG_INTERVAL_S = 2.0  # Debug log update interval
    PLOT_DISABLED_INTERVAL_MS = 100  # Update interval when plotting disabled
    SLIDER_RESOLUTION = 100  # Integer slider resolution multiplier

    # Plot control settings
    DEFAULT_PLOT_RATE_HZ = 10  # Default plot refresh rate
    MIN_PLOT_RATE_HZ = 1  # Minimum plot refresh rate
    MAX_PLOT_RATE_HZ = 100  # Maximum plot refresh rate
    DEFAULT_PLOT_ENABLED = True  # Plot updates enabled by default


class SimulationConfig:
    """Configuration for simulation mode."""
    UPDATE_RATE_MS = 2  # Physics update interval (500 Hz)
    DEFAULT_SERVO_TAU = 0.05  # Servo time constant (seconds)
    DEFAULT_SERVO_DELAY = 0.04  # Servo command delay (seconds)
    DEFAULT_SERVO_MAX_VELOCITY = 545.0  # Maximum servo velocity (deg/s)
    PHYSICS_SUBSTEPS = 1  # Physics integration substeps per update

# ============================================================================
# PID CONTROLLER CONFIGURATION
# ============================================================================

class PIDConfig:
    """Default PID controller parameters."""

    # Simulation defaults (aggressive tuning for ideal conditions)
    SIM_DEFAULT_GAINS = {
        'kp': 3.0,
        'ki': 0.0,
        'kd': 3.0
    }

    # Hardware defaults (conservative tuning for real system)
    HW_DEFAULT_GAINS = {
        'kp': 1.0,
        'ki': 0.0,
        'kd': 4.0
    }

    # Scalar multiplier values for GUI sliders
    SCALAR_VALUES = [
        0.0000001, 0.000001, 0.00001, 0.0001,
        0.001, 0.01, 0.1, 1.0, 10.0, 100.0
    ]

    # Default scalar indices (index into SCALAR_VALUES)
    SIM_SCALAR_INDICES = {
        'kp': 3,  # 0.0001 (matches hardware)
        'ki': 3,  # 0.0001 (matches hardware)
        'kd': 3  # 0.0001 (matches hardware)
    }

    HW_SCALAR_INDICES = {
        'kp': 6,  # 0.1
        'ki': 6,  # 0.1
        'kd': 5  # 0.01
    }

    # Controller limits
    OUTPUT_LIMIT = MAX_TILT_ANGLE_DEG  # Maximum controller output (degrees)
    INTEGRAL_LIMIT = 100.0  # Anti-windup limit for integral term

    # Derivative filtering
    DERIVATIVE_FILTER_ALPHA = 0.0  # Low-pass filter coefficient (0=none, >0=filtering)
    HW_DERIVATIVE_FILTER_ALPHA = 0.1  # More filtering for hardware (noisy measurements)

class LQRConfig:
    """Default LQR controller parameters."""

    # Default weights (same for simulation and hardware)
    DEFAULT_WEIGHTS = {
        'Q_pos': 1.0,  # Position error weight
        'Q_vel': 1.0,  # Velocity error weight
        'R': 1.0       # Control effort weight
    }

    # Scalar multiplier values for GUI sliders
    SCALAR_VALUES = [
        0.0000001, 0.000001, 0.00001, 0.0001,
        0.001, 0.01, 0.1, 1.0, 10.0, 100.0
    ]

    # Default scalar indices (index into SCALAR_VALUES)
    DEFAULT_SCALAR_INDICES = {
        'Q_pos': 9,  # 100.0
        'Q_vel': 5,  # 0.01
        'R': 5       # 0.01
    }

    # Controller limits
    OUTPUT_LIMIT = MAX_TILT_ANGLE_DEG  # Maximum controller output (degrees)

class TrajectoryPatternConfig:
    """Trajectory pattern parameters for target tracking."""

    # Circle pattern
    CIRCLE_RADIUS_RANGE_MM = (10.0, 100.0)
    CIRCLE_DEFAULT_RADIUS_MM = 50.0
    CIRCLE_PERIOD_RANGE_S = (3.0, 30.0)
    CIRCLE_DEFAULT_PERIOD_S = 10.0

    # Figure-8 pattern
    FIGURE8_WIDTH_RANGE_MM = (10.0, 150.0)
    FIGURE8_DEFAULT_WIDTH_MM = 60.0
    FIGURE8_HEIGHT_RANGE_MM = (10.0, 100.0)
    FIGURE8_DEFAULT_HEIGHT_MM = 40.0
    FIGURE8_PERIOD_RANGE_S = (3.0, 30.0)
    FIGURE8_DEFAULT_PERIOD_S = 12.0

    # Star pattern
    STAR_RADIUS_RANGE_MM = (10.0, 100.0)
    STAR_DEFAULT_RADIUS_MM = 60.0
    STAR_PERIOD_RANGE_S = (3.0, 30.0)
    STAR_DEFAULT_PERIOD_S = 15.0

class ManualPoseControlConfig:
    """Manual 6-DOF pose control slider limits."""

    # Translation limits (mm)
    X_RANGE_MM = (-115.0, 115.0)
    X_DEFAULT_MM = 0.0
    X_RESOLUTION_MM = 0.1

    Y_RANGE_MM = (-115.0, 115.0)
    Y_DEFAULT_MM = 0.0
    Y_RESOLUTION_MM = 0.1

    Z_OFFSET_RANGE_MM = (-90.0, 90.0)  # Offset from home height
    Z_RESOLUTION_MM = 0.1

    # Rotation limits (degrees)
    RX_RANGE_DEG = (-45.0, 45.0)
    RX_DEFAULT_DEG = 0.0
    RX_RESOLUTION_DEG = 0.1

    RY_RANGE_DEG = (-45.0, 45.0)
    RY_DEFAULT_DEG = 0.0
    RY_RESOLUTION_DEG = 0.1

    RZ_RANGE_DEG = (-115.0, 115.0)
    RZ_DEFAULT_DEG = 0.0
    RZ_RESOLUTION_DEG = 0.1

class KalmanFilterConfig:
    """Ball position/velocity Kalman filter parameters."""

    # Process noise (Q matrix scaling)
    PROCESS_NOISE_RANGE = (0.01, 10.0)
    DEFAULT_PROCESS_NOISE = 1.0

    # Measurement noise (R matrix scaling)
    MEASUREMENT_NOISE_RANGE = (0.01, 10.0)
    DEFAULT_MEASUREMENT_NOISE = 1.0

class BallControlConfig:
    """Ball control parameters (reset, push, etc.)."""

    # Ball push velocity (m/s)
    PUSH_VELOCITY_MS = 0.05  # Random velocity range: ±PUSH_VELOCITY_MS

# ============================================================================
# IK Z OPTIMIZATION CONFIGURATION
# ============================================================================

class IKZOptimizationConfig:
    """Configuration for dynamic Z offset optimization."""

    # Enable/disable flag (default off)
    ENABLED = False

    # Optimization parameters
    Z_SEARCH_RANGE_MM = 80.0  # Maximum Z adjustment range (±mm)
    MAX_ITERATIONS = 25  # Maximum optimization iterations
    TOLERANCE_DEG = 0.1  # Convergence tolerance (degrees)

# ============================================================================
# PIXY2 CAMERA CONFIGURATION
# ============================================================================

class Pixy2CameraConfig:
    """Pixy2 camera model parameters (based on measured hardware behavior)."""

    # Camera resolution (same for both V1 and V2)
    RESOLUTION_WIDTH_PX = 316  # Camera width (pixels)
    RESOLUTION_HEIGHT_PX = 208  # Camera height (pixels)

    # Camera center point in pixels (same for both V1 and V2)
    CENTER_X_PX = RESOLUTION_WIDTH_PX / 2.0  # 158 pixels
    CENTER_Y_PX = RESOLUTION_HEIGHT_PX / 2.0  # 104 pixels

    # Platform version-specific camera calibration
    if PLATFORM_VERSION == 'V1':
        # V1 Camera (original mounting and calibration)
        PIXEL_SIZE_MM = 1.4  # Physical size of one pixel (mm)
        SUBPIXEL_NOISE_STD_MM = 0.4  # Sub-pixel noise std dev (mm)
        FOV_WIDTH_MM = 350.0  # Physical width of camera view (mm)
        FOV_HEIGHT_MM = 266.0  # Physical height of camera view (mm)
        CENTER_X = 158.0  # Platform center X position in pixels (from control_v1.ino ORIGIN_X)
        CENTER_Y = 104.0  # Platform center Y position in pixels (from control_v1.ino ORIGIN_Y)
    else:  # V2
        # V2 Camera (repositioned and recalibrated)
        PIXEL_SIZE_MM = 2.0  # Physical size of one pixel (mm)
        SUBPIXEL_NOISE_STD_MM = 1.0  # Sub-pixel noise std dev (mm)
        FOV_WIDTH_MM = 558.0  # Physical width of camera view (mm) - calibrated
        FOV_HEIGHT_MM = 424.0  # Physical height of camera view (mm) - calibrated
        CENTER_X = 145.0  # Platform center X position offset (pixels + mm offset combined)
        CENTER_Y = 109.0  # Platform center Y position offset (pixels + mm offset combined)

    # Computed pixel-to-mm conversion factors (calculated from FOV and resolution)
    PIXELS_TO_MM_X = FOV_WIDTH_MM / RESOLUTION_WIDTH_PX
    PIXELS_TO_MM_Y = FOV_HEIGHT_MM / RESOLUTION_HEIGHT_PX

    # Default operational parameters
    DEFAULT_DETECTION_RATE = 0.999  # Ball detection probability (99.9%)
    DEFAULT_SAMPLE_RATE_HZ = 19.3  # Camera update rate (Hz)

    # GUI slider ranges
    PIXEL_SIZE_RANGE = (0.5, 3.0)  # mm
    NOISE_RANGE = (0.0, 2.0)  # mm (extended range for testing)  #1.0
    DETECTION_RATE_RANGE = (0.90, 1.0)  # probability
    SAMPLE_RATE_RANGE = (0.0, 60.0)  # Hz (0 = every frame)

# ============================================================================
# BALL PHYSICS CONFIGURATION
# ============================================================================

class BallPhysicsConfig:
    """Ball physics parameters."""

    # Ball properties
    RADIUS_M = 0.02  # Ball radius (meters) - 20mm ping pong ball
    MASS_KG = 0.0027  # Ball mass (kg) - 2.7g ping pong ball

    # Environmental parameters
    GRAVITY_M_S2 = 9.81  # Gravitational acceleration (m/s²)
    AIR_DENSITY_KG_M3 = 1.225  # Air density at sea level (kg/m³)
    DRAG_COEFFICIENT = 0.47  # Drag coefficient for sphere

    # Rolling dynamics
    ROLLING_FRICTION = 0.01  # Rolling resistance coefficient (measured from hardware)  #0.0225
    SPHERE_TYPE = 'hollow'  # 'hollow' or 'solid'

    # Moment of inertia factor (computed from sphere type)
    @classmethod
    def get_mass_factor(cls):
        """Calculate mass factor for rolling motion."""
        if cls.SPHERE_TYPE == 'solid':
            I_factor = 2.0 / 5.0
        elif cls.SPHERE_TYPE == 'hollow':
            I_factor = 2.0 / 3.0
        else:
            I_factor = 2.0 / 3.0  # Default to hollow

        # mass_factor = 1 + I/(m*r²)
        return 1.0 + I_factor

    # Get as dictionary for easy passing (for controllers: KalmanFilter, LQRController)
    @classmethod
    def as_dict(cls):
        """Return ball physics parameters for controllers (KalmanFilter, LQRController)."""
        return {
            'radius': cls.RADIUS_M,
            'mass': cls.MASS_KG,
            'gravity': cls.GRAVITY_M_S2,
            'mass_factor': cls.get_mass_factor(),
            'rolling_friction': cls.ROLLING_FRICTION,
            'sphere_type': cls.SPHERE_TYPE,
            'air_density': cls.AIR_DENSITY_KG_M3,
            'drag_coefficient': cls.DRAG_COEFFICIENT
        }

    @classmethod
    def for_physics_sim(cls):
        """Return ball physics parameters for SimpleBallPhysics2D (different parameter names)."""
        return {
            'ball_radius': cls.RADIUS_M,
            'ball_mass': cls.MASS_KG,
            'gravity': cls.GRAVITY_M_S2,
            'rolling_friction': cls.ROLLING_FRICTION,
            'sphere_type': cls.SPHERE_TYPE,
            'air_density': cls.AIR_DENSITY_KG_M3,
            'drag_coefficient': cls.DRAG_COEFFICIENT
        }

# ============================================================================
# SERIAL COMMUNICATION CONFIGURATION
# ============================================================================

class SerialConfig:
    """Serial communication parameters for hardware."""

    # Baud rates (optimized for performance)
    USB_BAUD_RATE = 200000  # USB serial to Arduino (200 kbps)
    MAESTRO_BAUD_RATE = 250000  # Maestro servo controller (250 kbps)

    # Timeouts
    READ_TIMEOUT_S = 0.1  # Serial read timeout (seconds)
    WRITE_TIMEOUT_S = 0.5  # Serial write timeout (seconds)

    # Connection parameters
    CONNECTION_DELAY_S = 2.0  # Delay after opening serial port (seconds)
    RECONNECT_DELAY_S = 0.1  # Delay between reconnection attempts (seconds)

    # Queue sizes
    BALL_DATA_QUEUE_SIZE = 10  # Camera data queue depth
    COMMAND_QUEUE_SIZE = 20  # Servo command queue depth
    IMU_GYRO_QUEUE_SIZE = 1000  # Gyroscope data queue depth
    IMU_ACCEL_QUEUE_SIZE = 2000  # Accelerometer data queue depth
    IMU_MAG_QUEUE_SIZE = 500  # Magnetometer data queue depth

    # Rate limiting
    MIN_COMMAND_INTERVAL_S = ControlLoopConfig.INTERVAL_S  # Minimum time between commands
    COMMAND_QUEUE_THRESHOLD_HIGH = 15  # Queue size for aggressive rate limiting
    COMMAND_QUEUE_THRESHOLD_MEDIUM = 10  # Queue size for medium rate limiting
    COMMAND_QUEUE_THRESHOLD_LOW = 5  # Queue size for normal operation

    # Rate limit intervals based on queue size
    RATE_LIMIT_HIGH_S = 0.05  # 20 Hz when queue is very full
    RATE_LIMIT_MEDIUM_S = 0.02  # 50 Hz when queue is moderately full
    RATE_LIMIT_NORMAL_S = MIN_COMMAND_INTERVAL_S  # 100 Hz in normal operation

    # Thread management
    THREAD_JOIN_TIMEOUT_S = 1.0  # Timeout for joining threads on shutdown
    READ_LOOP_SLEEP_S = 0.0005  # Sleep duration in read loop (0.5ms)
    WRITE_DELAY_S = 0.003  # Delay after each write in write loop (3ms)

# ============================================================================
# PERFORMANCE OPTIMIZATION CONFIGURATION
# ============================================================================

class PerformanceConfig:
    """Performance optimization parameters."""

    # IK cache settings
    IK_CACHE_SIZE = 5000  # Maximum cached IK solutions
    IK_CACHE_RESOLUTION_MM = 1.0  # Cache key resolution (mm/deg)

    # IK pre-warming
    IK_PREWARM_TILT_RANGE = (-15, 16)  # Tilt angles to pre-compute (degrees)
    IK_PREWARM_TILT_STEP = 2  # Step size for pre-warming (degrees)

    # Thread priorities (Windows)
    THREAD_PRIORITY_IDLE = -15
    THREAD_PRIORITY_LOWEST = -2
    THREAD_PRIORITY_BELOW_NORMAL = -1
    THREAD_PRIORITY_NORMAL = 0
    THREAD_PRIORITY_ABOVE_NORMAL = 1
    THREAD_PRIORITY_HIGHEST = 2
    THREAD_PRIORITY_TIME_CRITICAL = 15

    # Default control thread priority
    CONTROL_THREAD_PRIORITY = THREAD_PRIORITY_TIME_CRITICAL

    # Windows timer resolution
    TIMER_RESOLUTION_MS = 1  # Windows multimedia timer resolution (1ms)

    # Servo optimization
    SERVO_ANGLE_CHANGE_THRESHOLD = 0.2  # Minimum angle change to send command (degrees)

    # Timing statistics
    TIMING_STATS_MAX_SAMPLES = 1000  # Maximum timing samples to keep
    TIMING_BREAKPOINT_MAX_SAMPLES = 1000  # Maximum breakpoint timing samples

# ============================================================================
# IMU KALMAN FILTER CONFIGURATION
# ============================================================================

class IMUKalmanConfig:
    """IMU orientation Kalman filter parameters."""

    # Yaw tracking configuration
    ENABLE_YAW_TRACKING = False  # Enable full 6-DOF with magnetometer yaw tracking
    DEFAULT_MAG_NOISE = 0.1  # Magnetometer measurement noise (rad)

    # Kalman filter noise parameters
    DEFAULT_ACCEL_NOISE = 1.0
    DEFAULT_GYRO_NOISE = 0.0224
    DEFAULT_PROCESS_NOISE_ANGLE = 0.001
    DEFAULT_PROCESS_NOISE_BIAS = 0.001

    # Calibrated gyroscope bias (rad/s)
    CALIBRATED_GYRO_BIAS_X = 0.112679
    CALIBRATED_GYRO_BIAS_Y = 0.031500
    CALIBRATED_GYRO_BIAS_Z = 0.0  # Z-axis bias (for yaw tracking)

    # Magnetometer calibration
    MAG_OFFSET_X = 0.0  # Hard-iron offset X
    MAG_OFFSET_Y = 0.0  # Hard-iron offset Y
    MAG_OFFSET_Z = 0.0  # Hard-iron offset Z
    MAG_INCLINATION_DEG = 74.0  # Magnetic inclination angle (Norway, Ålesund)

    # Motion detection thresholds
    DEFAULT_ACCEL_THRESHOLD = 1.0  # m/s² - reject accel updates above this
    DEFAULT_GYRO_THRESHOLD = 0.5   # rad/s - gyro magnitude threshold

    # Scaling and transformation
    DEFAULT_GYRO_SCALE_MULTIPLIER = 1.0
    DEFAULT_MAG_INCLINATION_DEG = 74.283  # Magnetic inclination for Trondheim, Norway (74° 17')

    # Default axis transformations (identity - no flipping)
    DEFAULT_ACCEL_AXIS_FLIP = np.array([1, 1, 1])
    DEFAULT_GYRO_AXIS_FLIP = np.array([1, 1, 1])
    DEFAULT_ACCEL_ROTATION = np.eye(3)
    DEFAULT_GYRO_ROTATION = np.eye(3)

# ============================================================================
# IMU CALIBRATION CONFIGURATION
# ============================================================================

class IMUCalibrationConfig:
    """IMU calibration sequence timing parameters."""

    INITIALIZATION_DURATION_S = 3.0   # Stabilization period before calibration
    CALIBRATION_DURATION_S = 10.0     # Duration to collect calibration data

# ============================================================================
# HARDWARE CONNECTION CONFIGURATION
# ============================================================================

class HardwareConnectionConfig:
    """Serial connection setup timing and delays."""

    POST_CONNECTION_DELAY_S = 0.5      # Delay after establishing connection
    POST_SERVO_SPEED_DELAY_S = 0.1     # Delay after setting servo speed
    POST_SERVO_ACCEL_DELAY_S = 0.2     # Delay after setting servo acceleration

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

class VisualizationConfig:
    """Visualization and display parameters."""

    BALL_TRAIL_MAX_HISTORY = 200  # Maximum number of ball positions to track

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_vector_2d(vec, units="mm", decimals=1):
    """Format 2D vector for display."""
    return f"({vec[0]:.{decimals}f}, {vec[1]:.{decimals}f}) {units}"


def format_time(seconds, decimals=2):
    """Format time with consistent precision."""
    return f"{seconds:.{decimals}f}s"


def format_error_context(sim_time, ball_pos, ball_vel, error_msg):
    """Format error message with full context."""
    return (
        f"Error at t={format_time(sim_time)}: {error_msg}\n"
        f"Ball state: pos={format_vector_2d(ball_pos[:2])}, "
        f"vel={format_vector_2d(ball_vel[:2], 'mm/s')}"
    )

# ============================================================================
# CONFIGURATION UTILITY FUNCTIONS
# ============================================================================

def get_controller_defaults(controller_type='PID', mode='simulation'):
    """
    Get default controller parameters.

    Args:
        controller_type: 'PID' or 'LQR'
        mode: 'simulation' or 'hardware'

    Returns:
        Dictionary of default parameters
    """
    if controller_type.upper() == 'PID':
        if mode == 'simulation':
            return {
                'gains': PIDConfig.SIM_DEFAULT_GAINS.copy(),
                'scalar_indices': PIDConfig.SIM_SCALAR_INDICES.copy(),
                'scalar_values': PIDConfig.SCALAR_VALUES.copy(),
                'output_limit': PIDConfig.OUTPUT_LIMIT,
                'integral_limit': PIDConfig.INTEGRAL_LIMIT,
                'derivative_filter': PIDConfig.DERIVATIVE_FILTER_ALPHA
            }
        else:  # hardware
            return {
                'gains': PIDConfig.HW_DEFAULT_GAINS.copy(),
                'scalar_indices': PIDConfig.HW_SCALAR_INDICES.copy(),
                'scalar_values': PIDConfig.SCALAR_VALUES.copy(),
                'output_limit': PIDConfig.OUTPUT_LIMIT,
                'integral_limit': PIDConfig.INTEGRAL_LIMIT,
                'derivative_filter': PIDConfig.HW_DERIVATIVE_FILTER_ALPHA
            }

    elif controller_type.upper() == 'LQR':
        # LQR defaults (same for sim and hardware)
        return {
            'weights': LQRConfig.DEFAULT_WEIGHTS.copy(),
            'scalar_indices': LQRConfig.DEFAULT_SCALAR_INDICES.copy(),
            'scalar_values': LQRConfig.SCALAR_VALUES.copy(),
            'output_limit': LQRConfig.OUTPUT_LIMIT,
            'ball_physics': BallPhysicsConfig.as_dict()
        }

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")