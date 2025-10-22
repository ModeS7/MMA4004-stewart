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

# ============================================================================
# PHYSICAL CONSTANTS AND LIMITS
# ============================================================================

MAX_TILT_ANGLE_DEG = 15.0  # Maximum platform tilt magnitude (degrees)
MAX_SERVO_ANGLE_DEG = 40.0  # Maximum individual servo angle (degrees)
PLATFORM_SIZE_MM = 200.0  # Platform edge length (mm)
PLATFORM_HALF_SIZE_MM = 100.0  # Half platform size for boundary checks (mm)


# ============================================================================
# CONTROL LOOP CONFIGURATION
# ============================================================================

class ControlLoopConfig:
    """Configuration for real-time control loop (hardware mode)."""
    FREQUENCY_HZ = 100  # Control loop frequency
    INTERVAL_S = 1.0 / FREQUENCY_HZ  # Control loop period (10ms)
    IK_TIMEOUT_S = 0.008  # Maximum time allowed for IK calculation (8ms)
    MAX_LOOP_TIME_S = 0.01  # Maximum acceptable loop time (10ms)


class GUIConfig:
    """Configuration for GUI updates (applies to both sim and hardware)."""
    UPDATE_HZ = 5  # GUI refresh rate
    UPDATE_INTERVAL_MS = 200  # GUI update interval (milliseconds)
    LOG_INTERVAL_S = 2.0  # Debug log update interval


class SimulationConfig:
    """Configuration for simulation mode."""
    UPDATE_RATE_MS = 20  # Physics update interval (50 Hz)
    DEFAULT_SERVO_TAU = 0.1  # Servo time constant (seconds)
    DEFAULT_SERVO_DELAY = 0.0  # Servo command delay (seconds)
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
        'ki': 1.0,
        'kd': 3.0
    }

    # Hardware defaults (conservative tuning for real system)
    HW_DEFAULT_GAINS = {
        'kp': 1.0,
        'ki': 1.0,
        'kd': 4.0
    }

    # Scalar multiplier values for GUI sliders
    SCALAR_VALUES = [
        0.0000001, 0.000001, 0.00001, 0.0001,
        0.001, 0.01, 0.1, 1.0, 10.0, 100.0
    ]

    # Default scalar indices (index into SCALAR_VALUES)
    SIM_SCALAR_INDICES = {
        'kp': 4,  # 0.001
        'ki': 4,  # 0.001
        'kd': 4  # 0.001
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


# ============================================================================
# LQR CONTROLLER CONFIGURATION
# ============================================================================

class LQRConfig:
    """Default LQR controller parameters."""

    # Simulation defaults
    SIM_DEFAULT_WEIGHTS = {
        'Q_pos': 1.0,
        'Q_vel': 1.0,
        'R': 1.0
    }

    # Hardware defaults (more aggressive position control)
    HW_DEFAULT_WEIGHTS = {
        'Q_pos': 1.0,
        'Q_vel': 1.0,
        'R': 1.0
    }

    # Scalar multiplier values for GUI sliders
    SCALAR_VALUES = [
        0.0000001, 0.000001, 0.00001, 0.0001,
        0.001, 0.01, 0.1, 1.0, 10.0, 100.0
    ]

    # Default scalar indices
    SIM_SCALAR_INDICES = {
        'Q_pos': 7,  # 1.0
        'Q_vel': 6,  # 0.1
        'R': 5  # 0.01
    }

    HW_SCALAR_INDICES = {
        'Q_pos': 7,  # 1.0
        'Q_vel': 5,  # 0.01
        'R': 5  # 0.01
    }

    # Controller limits
    OUTPUT_LIMIT = MAX_TILT_ANGLE_DEG  # Maximum controller output (degrees)


# ============================================================================
# KALMAN FILTER CONFIGURATION
# ============================================================================

class KalmanFilterConfig:
    """Default Kalman filter parameters."""

    # Noise scaling factors (tunable multipliers)
    DEFAULT_PROCESS_NOISE_SCALE = 1.0  # Trust model vs measurements
    DEFAULT_MEASUREMENT_NOISE_SCALE = 1.0  # Trust measurements vs smoothing

    # Base noise covariances (before scaling)
    # These are physical estimates based on system characteristics
    BASE_PROCESS_NOISE = {
        'position': 0.0001,  # Position uncertainty (m²)
        'velocity': 0.001  # Velocity uncertainty (m²/s²)
    }

    # Camera measurement noise (based on Pixy2 characteristics)
    CAMERA_PIXEL_SIZE = 0.0014  # 1.4mm in meters
    CAMERA_SUBPIXEL_NOISE = 0.0004  # 0.4mm std dev in meters

    # Computed measurement variance (quantization + subpixel noise)
    @classmethod
    def get_measurement_variance(cls):
        """Calculate measurement noise variance."""
        quantization_var = (cls.CAMERA_PIXEL_SIZE ** 2) / 12
        subpixel_var = cls.CAMERA_SUBPIXEL_NOISE ** 2
        return quantization_var + subpixel_var

    # Filter initialization
    INITIAL_COVARIANCE = 0.01  # Initial state uncertainty

    # GUI slider ranges
    NOISE_SCALE_MIN = 0.01
    NOISE_SCALE_MAX = 10.0


# ============================================================================
# PIXY2 CAMERA CONFIGURATION
# ============================================================================

class Pixy2CameraConfig:
    """Pixy2 camera model parameters (based on measured hardware behavior)."""

    # Physical camera characteristics
    PIXEL_SIZE_MM = 1.4  # Physical size of one pixel (mm)
    SUBPIXEL_NOISE_STD_MM = 0.4  # Sub-pixel noise std dev (mm)

    # Field of view dimensions
    FOV_WIDTH_MM = 350.0  # Physical width of camera view (mm)
    FOV_HEIGHT_MM = 266.0  # Physical height of camera view (mm)

    # Camera resolution
    RESOLUTION_WIDTH_PX = 316  # Camera width (pixels)
    RESOLUTION_HEIGHT_PX = 208  # Camera height (pixels)

    # Computed pixel-to-mm conversion factors
    PIXELS_TO_MM_X = FOV_WIDTH_MM / RESOLUTION_WIDTH_PX
    PIXELS_TO_MM_Y = FOV_HEIGHT_MM / RESOLUTION_HEIGHT_PX

    # Camera center point (for coordinate transformation)
    CENTER_X_PX = RESOLUTION_WIDTH_PX / 2.0  # 158
    CENTER_Y_PX = RESOLUTION_HEIGHT_PX / 2.0  # 104

    # Detection characteristics
    DEFAULT_DETECTION_RATE = 0.999  # 99.9% detection rate (very reliable)
    DEFAULT_SAMPLE_RATE_HZ = 19.3  # Measured camera update rate (Hz)

    # GUI slider ranges
    PIXEL_SIZE_RANGE = (0.5, 3.0)  # mm
    NOISE_RANGE = (0.0, 1.0)  # mm
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
    ROLLING_FRICTION = 0.0225  # Rolling resistance coefficient
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

    # Get as dictionary for easy passing
    @classmethod
    def as_dict(cls):
        """Return ball physics parameters as dictionary."""
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


# ============================================================================
# STEWART PLATFORM GEOMETRY CONFIGURATION
# ============================================================================

class StewartPlatformConfig:
    """Stewart platform geometric parameters."""

    # Link lengths
    HORN_LENGTH_MM = 31.75  # Servo horn length (mm)
    ROD_LENGTH_MM = 145.0  # Push rod length (mm)

    # Base geometry
    BASE_RADIUS_MM = 73.025  # Base circle radius (mm)
    BASE_ANCHORS_OFFSET_MM = 36.8893  # Anchor offset from base circle (mm)

    # Platform geometry
    PLATFORM_RADIUS_MM = 67.775  # Platform circle radius (mm)
    PLATFORM_ANCHORS_OFFSET_MM = 12.7  # Anchor offset from platform circle (mm)

    # Vertical offset
    TOP_SURFACE_OFFSET_MM = 26.0  # Distance from platform anchors to top surface (mm)

    # Computed home positions (calculated in StewartPlatformIK.__init__)
    # HOME_HEIGHT_MM - computed from geometry
    # HOME_HEIGHT_TOP_SURFACE_MM - home_height + top_surface_offset

    @classmethod
    def as_dict(cls):
        """Return platform geometry as dictionary."""
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

    # Rate limiting
    MIN_COMMAND_INTERVAL_S = ControlLoopConfig.INTERVAL_S  # Minimum time between commands
    COMMAND_QUEUE_THRESHOLD_HIGH = 15  # Queue size for aggressive rate limiting
    COMMAND_QUEUE_THRESHOLD_MEDIUM = 10  # Queue size for medium rate limiting
    COMMAND_QUEUE_THRESHOLD_LOW = 5  # Queue size for normal operation

    # Rate limit intervals based on queue size
    RATE_LIMIT_HIGH_S = 0.05  # 20 Hz when queue is very full
    RATE_LIMIT_MEDIUM_S = 0.02  # 50 Hz when queue is moderately full
    RATE_LIMIT_NORMAL_S = MIN_COMMAND_INTERVAL_S  # 100 Hz in normal operation


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

    # Debug logging
    DEBUG_LOG_INTERVAL_LOOPS = 50  # Log control values every N loops (0.5s at 100Hz)


# ============================================================================
# TRAJECTORY PATTERN CONFIGURATION
# ============================================================================

class TrajectoryPatternConfig:
    """Default parameters for trajectory patterns."""

    # Circle pattern
    CIRCLE_RADIUS_MM = 50.0
    CIRCLE_PERIOD_S = 10.0
    CIRCLE_CLOCKWISE = True

    # Figure-8 pattern
    FIGURE8_WIDTH_MM = 60.0
    FIGURE8_HEIGHT_MM = 40.0
    FIGURE8_PERIOD_S = 12.0

    # Star pattern
    STAR_RADIUS_MM = 60.0
    STAR_PERIOD_S = 15.0
    STAR_NUM_POINTS = 5

    # GUI slider ranges
    RADIUS_RANGE = (10.0, 100.0)  # mm
    WIDTH_RANGE = (10.0, 150.0)  # mm
    HEIGHT_RANGE = (10.0, 100.0)  # mm
    PERIOD_RANGE = (3.0, 30.0)  # seconds


# ============================================================================
# GUI COLOR SCHEME
# ============================================================================

class ColorScheme:
    """Dark theme color scheme for GUI."""

    BG = '#1e1e1e'  # Main background
    PANEL_BG = '#2d2d2d'  # Panel background
    WIDGET_BG = '#3d3d3d'  # Widget background
    FG = '#e0e0e0'  # Foreground text
    HIGHLIGHT = '#007acc'  # Highlight/accent color
    BUTTON_BG = '#0e639c'  # Button background
    BUTTON_FG = '#ffffff'  # Button foreground
    ENTRY_BG = '#3d3d3d'  # Entry widget background
    BORDER = '#555555'  # Border color
    SUCCESS = '#4ec9b0'  # Success/active indicator
    WARNING = '#ce9178'  # Warning indicator
    ERROR = '#f44747'  # Error indicator

    @classmethod
    def as_dict(cls):
        """Return color scheme as dictionary."""
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
            'warning': cls.WARNING,
            'error': cls.ERROR
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_vector_2d(vec, units="mm", decimals=1):
    """Format 2D vector for display."""
    return f"({vec[0]:.{decimals}f}, {vec[1]:.{decimals}f}) {units}"


def format_time(seconds, decimals=2):
    """Format time with consistent precision."""
    return f"{seconds:.{decimals}f}s"


def format_angle(degrees, decimals=2):
    """Format angle with degree symbol."""
    return f"{degrees:.{decimals}f}°"


def format_error_context(sim_time, ball_pos, ball_vel, error_msg):
    """Format error message with full context."""
    return (
        f"Error at t={format_time(sim_time)}: {error_msg}\n"
        f"Ball state: pos={format_vector_2d(ball_pos[:2])}, "
        f"vel={format_vector_2d(ball_vel[:2], 'mm/s')}"
    )


def is_position_in_bounds(x, y, max_size=PLATFORM_HALF_SIZE_MM):
    """Check if position is within platform bounds."""
    return abs(x) <= max_size and abs(y) <= max_size


def is_tilt_magnitude_valid(rx, ry, max_tilt=MAX_TILT_ANGLE_DEG):
    """Check if tilt magnitude is within limits."""
    import numpy as np
    magnitude = np.sqrt(rx ** 2 + ry ** 2)
    return magnitude <= max_tilt


def is_servo_angle_valid(angle, max_angle=MAX_SERVO_ANGLE_DEG):
    """Check if servo angle is within limits."""
    return abs(angle) <= max_angle


# ============================================================================
# CONFIGURATION SUMMARY FUNCTIONS
# ============================================================================

def print_config_summary():
    """Print summary of all configuration parameters."""
    print("=" * 70)
    print("STEWART PLATFORM CONFIGURATION SUMMARY")
    print("=" * 70)
    print()

    print("PHYSICAL LIMITS:")
    print(f"  Max Tilt: {MAX_TILT_ANGLE_DEG}°")
    print(f"  Max Servo: {MAX_SERVO_ANGLE_DEG}°")
    print(f"  Platform Size: {PLATFORM_SIZE_MM}mm")
    print()

    print("CONTROL LOOP:")
    print(f"  Frequency: {ControlLoopConfig.FREQUENCY_HZ} Hz")
    print(f"  Period: {ControlLoopConfig.INTERVAL_S * 1000:.1f} ms")
    print()

    print("PID DEFAULTS (Simulation):")
    print(f"  Kp: {PIDConfig.SIM_DEFAULT_GAINS['kp']}")
    print(f"  Ki: {PIDConfig.SIM_DEFAULT_GAINS['ki']}")
    print(f"  Kd: {PIDConfig.SIM_DEFAULT_GAINS['kd']}")
    print()

    print("PID DEFAULTS (Hardware):")
    print(f"  Kp: {PIDConfig.HW_DEFAULT_GAINS['kp']}")
    print(f"  Ki: {PIDConfig.HW_DEFAULT_GAINS['ki']}")
    print(f"  Kd: {PIDConfig.HW_DEFAULT_GAINS['kd']}")
    print()

    print("LQR DEFAULTS (Simulation):")
    print(f"  Q_pos: {LQRConfig.SIM_DEFAULT_WEIGHTS['Q_pos']}")
    print(f"  Q_vel: {LQRConfig.SIM_DEFAULT_WEIGHTS['Q_vel']}")
    print(f"  R: {LQRConfig.SIM_DEFAULT_WEIGHTS['R']}")
    print()

    print("PIXY2 CAMERA:")
    print(f"  Resolution: {Pixy2CameraConfig.RESOLUTION_WIDTH_PX}×{Pixy2CameraConfig.RESOLUTION_HEIGHT_PX} px")
    print(f"  FOV: {Pixy2CameraConfig.FOV_WIDTH_MM}×{Pixy2CameraConfig.FOV_HEIGHT_MM} mm")
    print(f"  Pixel Size: {Pixy2CameraConfig.PIXEL_SIZE_MM} mm")
    print(f"  Sample Rate: {Pixy2CameraConfig.DEFAULT_SAMPLE_RATE_HZ} Hz")
    print()

    print("BALL PHYSICS:")
    print(f"  Radius: {BallPhysicsConfig.RADIUS_M * 1000} mm")
    print(f"  Mass: {BallPhysicsConfig.MASS_KG * 1000} g")
    print(f"  Type: {BallPhysicsConfig.SPHERE_TYPE}")
    print(f"  Mass Factor: {BallPhysicsConfig.get_mass_factor():.3f}")
    print()

    print("SERIAL COMMUNICATION:")
    print(f"  USB Baud: {SerialConfig.USB_BAUD_RATE}")
    print(f"  Maestro Baud: {SerialConfig.MAESTRO_BAUD_RATE}")
    print()

    print("PERFORMANCE:")
    print(f"  IK Cache Size: {PerformanceConfig.IK_CACHE_SIZE}")
    print(f"  Thread Priority: {PerformanceConfig.CONTROL_THREAD_PRIORITY}")
    print(f"  Timer Resolution: {PerformanceConfig.TIMER_RESOLUTION_MS} ms")
    print()

    print("=" * 70)


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
        if mode == 'simulation':
            return {
                'weights': LQRConfig.SIM_DEFAULT_WEIGHTS.copy(),
                'scalar_indices': LQRConfig.SIM_SCALAR_INDICES.copy(),
                'scalar_values': LQRConfig.SCALAR_VALUES.copy(),
                'output_limit': LQRConfig.OUTPUT_LIMIT,
                'ball_physics': BallPhysicsConfig.as_dict()
            }
        else:  # hardware
            return {
                'weights': LQRConfig.HW_DEFAULT_WEIGHTS.copy(),
                'scalar_indices': LQRConfig.HW_SCALAR_INDICES.copy(),
                'scalar_values': LQRConfig.SCALAR_VALUES.copy(),
                'output_limit': LQRConfig.OUTPUT_LIMIT,
                'ball_physics': BallPhysicsConfig.as_dict()
            }

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


# ============================================================================
# MAIN - PRINT CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    print_config_summary()