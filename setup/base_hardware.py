#!/usr/bin/env python3
"""
Hardware Controller Base Class

Base class for hardware-enabled Stewart Platform controllers.
Provides shared hardware communication, control loop, and Kalman filtering logic.

This base class eliminates code duplication between StewartController (full_c.py)
and MinimalController (min_c.py) while allowing each to have unique GUIs.
"""

import gc
import numpy as np
import time
import threading
import serial
from queue import Queue, Empty
import sys
import ctypes

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QWidget, QApplication

from setup.base_simulator import BaseStewartSimulator, ControllerConfig
from core.control_core import PIDController, LQRController, KalmanFilter, clip_tilt_vector
from core.utils import (ControlLoopConfig, Pixy2CameraConfig, BallPhysicsConfig,
                         VisualizationConfig, HardwareConnectionConfig, PIDConfig,
                         PerformanceConfig, IKZOptimizationConfig, get_controller_defaults,
                         SerialConfig)

# Thread priority constants
THREAD_PRIORITY_IDLE = -15
THREAD_PRIORITY_LOWEST = -2
THREAD_PRIORITY_BELOW_NORMAL = -1
THREAD_PRIORITY_NORMAL = 0
THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_HIGHEST = 2
THREAD_PRIORITY_TIME_CRITICAL = 15


# ============================================================================
# HARDWARE UTILITY CLASSES
# ============================================================================

class WindowsTimerManager:
    """Windows multimedia timer resolution manager."""

    def __init__(self):
        self.timer_set = False
        self.is_windows = sys.platform.startswith('win')

    def set_high_resolution(self):
        """Set Windows timer to high resolution."""
        if not self.is_windows:
            return False, "Not Windows - timer not set"

        try:
            timeBeginPeriod = ctypes.windll.winmm.timeBeginPeriod
            result = timeBeginPeriod(1)
            if result == 0:
                self.timer_set = True
                return True, "Windows timer resolution set"
            else:
                return False, f"Timer set failed: {result}"
        except Exception as e:
            return False, f"Timer error: {str(e)}"

    def restore_default(self):
        """Restore default timer resolution."""
        if self.timer_set:
            try:
                timeEndPeriod = ctypes.windll.winmm.timeEndPeriod
                timeEndPeriod(1)
                self.timer_set = False
            except:
                pass


class ThreadPriorityManager:
    """Windows thread priority manager."""

    def __init__(self):
        self.is_windows = sys.platform.startswith('win')
        self.kernel32 = None

        if self.is_windows:
            try:
                self.kernel32 = ctypes.windll.kernel32
            except (AttributeError, OSError):
                self.is_windows = False

    def set_thread_priority(self, thread_id, priority=THREAD_PRIORITY_ABOVE_NORMAL):
        """
        Set thread priority on Windows.

        Args:
            thread_id: Thread ID from thread.ident
            priority: Priority level (1=ABOVE_NORMAL, 2=HIGHEST)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_windows or self.kernel32 is None:
            return False

        try:
            handle = self.kernel32.OpenThread(0x0020, False, thread_id)
            if not handle:
                return False

            result = self.kernel32.SetThreadPriority(handle, priority)
            self.kernel32.CloseHandle(handle)

            return bool(result)
        except Exception:
            return False


class IKCache:
    """Cache IK results with coarse resolution for higher hit rate."""

    def __init__(self, max_size=5000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_key(self, translation, rotation):
        t_key = tuple(np.round(translation, 0))
        r_key = tuple(np.round(rotation, 0))
        return (t_key, r_key)

    def get(self, translation, rotation):
        key = self.get_key(translation, rotation)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, translation, rotation, angles):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        key = self.get_key(translation, rotation)
        self.cache[key] = angles.copy()

    def get_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class SerialController:
    """Serial communication with hardware."""

    def __init__(self, port, baudrate=SerialConfig.USB_BAUD_RATE):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        self.read_thread = None
        self.write_thread = None
        self.running = False

        self.ball_data_queue = Queue(maxsize=SerialConfig.BALL_DATA_QUEUE_SIZE)
        self.command_queue = Queue(maxsize=SerialConfig.COMMAND_QUEUE_SIZE)
        self.last_command_time = 0

        # IMU data queues
        self.gyro_queue = Queue(maxsize=SerialConfig.IMU_GYRO_QUEUE_SIZE)
        self.accel_queue = Queue(maxsize=SerialConfig.IMU_ACCEL_QUEUE_SIZE)
        self.mag_queue = Queue(maxsize=SerialConfig.IMU_MAG_QUEUE_SIZE)

        # IMU statistics
        self.gyro_count = 0
        self.accel_count = 0
        self.mag_count = 0

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate,
                                        timeout=SerialConfig.READ_TIMEOUT_S,
                                        write_timeout=SerialConfig.WRITE_TIMEOUT_S)
            time.sleep(SerialConfig.CONNECTION_DELAY_S)
            self.connected = True

            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            self.running = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()

            self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
            self.write_thread.start()

            return True, "Connected successfully"
        except Exception as e:
            self.connected = False
            return False, f"Connection failed: {str(e)}"

    def disconnect(self):
        self.running = False

        if self.read_thread:
            self.read_thread.join(timeout=1.0)
        if self.write_thread:
            self.write_thread.join(timeout=1.0)

        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False

    def _read_loop(self):
        buffer = ""

        while self.running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting > 0:
                    chunk = self.serial.read(self.serial.in_waiting).decode(
                        'utf-8', errors='ignore'
                    )
                    buffer += chunk

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if line.startswith("BALL:"):
                            try:
                                parts = line[5:].split(',')
                                if len(parts) == 6:
                                    ball_data = {
                                        'timestamp': float(parts[0]),
                                        'x': float(parts[1]),
                                        'y': float(parts[2]),
                                        'detected': bool(int(parts[3])),
                                        'error_x': float(parts[4]),
                                        'error_y': float(parts[5])
                                    }

                                    if self.ball_data_queue.full():
                                        try:
                                            self.ball_data_queue.get_nowait()
                                        except Empty:
                                            pass
                                    self.ball_data_queue.put(ball_data)
                            except (ValueError, IndexError):
                                pass

                        elif line.startswith("A:"):
                            # Accelerometer data: A:timestamp_us,ax,ay,az
                            try:
                                parts = line[2:].split(',')
                                if len(parts) == 4:
                                    timestamp_us = int(parts[0])
                                    ax, ay, az = int(parts[1]), int(parts[2]), int(parts[3])
                                    accel_data = np.array([ax, ay, az])

                                    if not self.accel_queue.full():
                                        self.accel_queue.put((timestamp_us, accel_data))
                                        self.accel_count += 1
                            except (ValueError, IndexError):
                                pass

                        elif line.startswith("G:"):
                            # Gyroscope data: G:timestamp_us,gx,gy,gz
                            try:
                                parts = line[2:].split(',')
                                if len(parts) == 4:
                                    timestamp_us = int(parts[0])
                                    gx, gy, gz = int(parts[1]), int(parts[2]), int(parts[3])
                                    gyro_data = np.array([gx, gy, gz])

                                    if not self.gyro_queue.full():
                                        self.gyro_queue.put((timestamp_us, gyro_data))
                                        self.gyro_count += 1
                            except (ValueError, IndexError):
                                pass

                        elif line.startswith("M:"):
                            # Magnetometer data: M:timestamp_us,mx,my,mz
                            try:
                                parts = line[2:].split(',')
                                if len(parts) == 4:
                                    timestamp_us = int(parts[0])
                                    mx, my, mz = int(parts[1]), int(parts[2]), int(parts[3])
                                    mag_data = np.array([mx, my, mz])

                                    if not self.mag_queue.full():
                                        self.mag_queue.put((timestamp_us, mag_data))
                                        self.mag_count += 1
                            except (ValueError, IndexError):
                                pass

                time.sleep(SerialConfig.READ_LOOP_SLEEP_S)

            except Exception as e:
                if self.running:
                    print(f"Serial read failed: {e}")
                time.sleep(0.1)

    def _write_loop(self):
        error_count = 0
        max_errors = 5

        while self.running and self.serial and self.serial.is_open:
            try:
                try:
                    command = self.command_queue.get(timeout=0.01)
                    self.serial.write(command.encode('utf-8'))
                    self.last_command_time = time.time()
                    error_count = 0
                    time.sleep(SerialConfig.WRITE_DELAY_S)

                except Empty:
                    pass

            except serial.SerialTimeoutException:
                error_count += 1
                if error_count > max_errors:
                    while not self.command_queue.empty():
                        try:
                            self.command_queue.get_nowait()
                        except Empty:
                            break
                    error_count = 0
                time.sleep(0.1)

            except Exception as e:
                if self.running:
                    error_count += 1
                time.sleep(0.1)

    def send_servo_angles(self, angles):
        """Queue servo angles (no rate limiting for high-frequency control)."""
        if not self.connected:
            return False

        try:
            command = ','.join([f'{angle:.2f}' for angle in angles]) + '\n'

            # Drop oldest commands if queue is full (keep system responsive)
            if self.command_queue.qsize() >= SerialConfig.COMMAND_QUEUE_SIZE:
                try:
                    self.command_queue.get_nowait()  # Remove oldest
                except:
                    pass

            self.command_queue.put_nowait(command)
            self.last_command_time = time.time()
            return True
        except Exception as e:
            print(f"Command queue failed: {e}")
            return False

    def send_command(self, cmd):
        if not self.connected:
            return False
        try:
            if self.command_queue.qsize() >= SerialConfig.COMMAND_QUEUE_THRESHOLD_HIGH:
                return False
            self.command_queue.put_nowait(cmd + '\n')
            return True
        except:
            return False

    def set_servo_speed(self, speed):
        return self.send_command(f'SPD:{speed}')

    def set_servo_acceleration(self, accel):
        return self.send_command(f'ACC:{accel}')

    def get_latest_ball_data(self):
        try:
            data = None
            while not self.ball_data_queue.empty():
                data = self.ball_data_queue.get_nowait()
            return data
        except Empty:
            return None

    def get_imu_data_batch(self):
        """Get all available IMU data from queues for calibration."""
        gyro_batch = []
        accel_batch = []
        mag_batch = []

        try:
            while not self.gyro_queue.empty():
                gyro_batch.append(self.gyro_queue.get_nowait())
        except Empty:
            pass

        try:
            while not self.accel_queue.empty():
                accel_batch.append(self.accel_queue.get_nowait())
        except Empty:
            pass

        try:
            while not self.mag_queue.empty():
                mag_batch.append(self.mag_queue.get_nowait())
        except Empty:
            pass

        return gyro_batch, accel_batch, mag_batch

    def get_single_imu_sample(self):
        """Get one sample from each IMU sensor (for real-time processing)."""
        gyro_data = None
        accel_data = None
        mag_data = None

        try:
            # Get latest from each queue
            while not self.gyro_queue.empty():
                gyro_data = self.gyro_queue.get_nowait()
        except Empty:
            pass

        try:
            while not self.accel_queue.empty():
                accel_data = self.accel_queue.get_nowait()
        except Empty:
            pass

        try:
            while not self.mag_queue.empty():
                mag_data = self.mag_queue.get_nowait()
        except Empty:
            pass

        return gyro_data, accel_data, mag_data


class HardwareControllerConfig(ControllerConfig):
    """Hardware PID controller configuration with derivative filtering."""

    def __init__(self):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0]
        self.default_gains = {'kp': 3.0, 'ki': 1.0, 'kd': 3.0}
        self.default_scalar_idx = 3

    def get_controller_name(self) -> str:
        return "PID (Hardware)"

    def create_controller(self, **kwargs):
        return PIDController(
            kp=kwargs.get('kp', 0.0003),
            ki=kwargs.get('ki', 0.0001),
            kd=kwargs.get('kd', 0.0003),
            output_limit=kwargs.get('output_limit', 15.0),
            derivative_filter_alpha=kwargs.get('derivative_filter_alpha', PIDConfig.HW_DERIVATIVE_FILTER_ALPHA)
        )

    def get_scalar_values(self) -> list:
        return self.scalar_values


class LQRControllerConfig(ControllerConfig):
    """LQR controller configuration for both simulation and hardware modes."""

    def __init__(self, mode='simulation'):
        """
        Initialize LQR configuration.

        Args:
            mode: 'simulation' or 'hardware'
        """
        config = get_controller_defaults('LQR', mode)
        self.scalar_values = config['scalar_values']
        self.default_weights = config['weights']
        self.default_scalar_indices = config['scalar_indices']
        self.output_limit = config['output_limit']
        self.ball_physics_params = BallPhysicsConfig.as_dict()
        self.controller_ref = None
        self.mode = mode

    def get_controller_name(self) -> str:
        return "LQR"

    def create_controller(self, **kwargs):
        return LQRController(
            Q_pos=kwargs.get('Q_pos', self.default_weights['Q_pos']),
            Q_vel=kwargs.get('Q_vel', self.default_weights['Q_vel']),
            R=kwargs.get('R', self.default_weights['R']),
            output_limit=kwargs.get('output_limit', self.output_limit),
            ball_physics_params=self.ball_physics_params
        )

    def get_scalar_values(self) -> list:
        return self.scalar_values


# ============================================================================
# HARDWARE CONTROLLER BASE CLASS
# ============================================================================


class HardwareControllerBase(BaseStewartSimulator):
    """Base class for hardware-enabled Stewart Platform controllers.

    Provides:
    - Hardware communication (serial, IMU, camera)
    - Control thread management
    - Kalman filtering for ball state
    - IK caching and pre-warming
    - Mode switching (simulation ↔ hardware)
    - Controller switching (PID ↔ LQR ↔ Manual)

    Subclasses must implement:
    - GUI layout configuration (number and type of modules)
    - Controller-specific widgets (if any)
    """

    def __init__(self, app, controller_config):
        """Initialize hardware controller base.

        Args:
            app: QApplication instance
            controller_config: ControllerConfig instance for this controller type
        """
        # Hardware components
        self.serial_controller = None
        self.connected = False
        self.port_var = ''
        self.ik_cache = None
        self.timer_manager = WindowsTimerManager()
        self.priority_manager = ThreadPriorityManager()
        self.control_frequency = ControlLoopConfig.DEFAULT_FREQUENCY_HZ
        self.control_interval = 1.0 / ControlLoopConfig.DEFAULT_FREQUENCY_HZ
        self.use_kalman_derivative = False

        # Camera calibration parameters (pixels to mm conversion)
        self.pixy_width_mm = Pixy2CameraConfig.FOV_WIDTH_MM
        self.pixy_height_mm = Pixy2CameraConfig.FOV_HEIGHT_MM
        self.pixels_to_mm_x = Pixy2CameraConfig.PIXELS_TO_MM_X
        self.pixels_to_mm_y = Pixy2CameraConfig.PIXELS_TO_MM_Y
        self.last_ball_update = 0.0
        self.ball_pos_mm = np.array([0.0, 0.0])
        self.ball_detected = False

        # Performance tracking
        self.performance_data = {
            'loop_times': [],
            'ik_times': [],
            'serial_times': []
        }

        # Initialize Kalman filter (required before super().__init__)
        self.kalman_filter = KalmanFilter(
            process_noise_scale=1.0,
            measurement_noise_scale=1.0,
            ball_physics_params=BallPhysicsConfig.as_dict(),
            dt=self.control_interval
        )
        self.kalman_enabled = True

        # Ball position trail visualization
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = VisualizationConfig.BALL_TRAIL_MAX_HISTORY

        # Call parent constructor
        super().__init__(app, controller_config)

    def _get_controller_type(self):
        """Override to return selected controller type."""
        return self.controller_type_selection

    def _create_controller_config(self):
        """Create appropriate controller config based on mode and controller type."""
        if self.controller_type_selection == 'PID':
            return HardwareControllerConfig()
        elif self.controller_type_selection == 'LQR':
            if self.operation_mode == 'real':
                return LQRControllerConfig(mode='hardware')
            else:
                return LQRControllerConfig(mode='simulation')
        else:
            # Manual mode uses default configuration
            return HardwareControllerConfig()

    def _initialize_controller(self):
        """Initialize controller based on current type."""
        if self.controller_type_selection == 'Manual':
            self.controller = None
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        # Validate GUI components availability
        controller_name = self.controller_config.get_controller_name()
        if "PID" in controller_name and 'kp' not in sliders:
            self.controller = None
            return
        if "LQR" in controller_name and 'Q_pos' not in sliders:
            self.controller = None
            return

        if "PID" in controller_name:
            kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
            ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
            kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

            self.controller = PIDController(
                kp=kp, ki=ki, kd=kd,
                output_limit=15.0,
                derivative_filter_alpha=PIDConfig.HW_DERIVATIVE_FILTER_ALPHA
            )
            self.log(f"PID initialized: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

        elif "LQR" in controller_name:
            Q_pos = self.controller_config.get_scaled_param('Q_pos', sliders, scalar_vars)
            Q_vel = self.controller_config.get_scaled_param('Q_vel', sliders, scalar_vars)
            R = self.controller_config.get_scaled_param('R', sliders, scalar_vars)

            try:
                self.controller = LQRController(
                    Q_pos=Q_pos,
                    Q_vel=Q_vel,
                    R=R,
                    output_limit=15.0
                )
                self.log(f"LQR initialized: Q_pos={Q_pos:.2e}, Q_vel={Q_vel:.2e}, R={R:.2e}")
            except Exception as e:
                self.log(f"LQR initialization failed: {str(e)}")
                self.log("Adjust Q/R parameters and retry")
                self.controller = None

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Update controller with current ball state."""
        if self.controller is None:
            return 0.0, 0.0

        if isinstance(self.controller, PIDController):
            error_x = target_pos_mm[0] - ball_pos_mm[0]
            error_y = target_pos_mm[1] - ball_pos_mm[1]

            rx, ry = self.controller.update(error_x, error_y, dt,
                                           ball_vel_mm_s=(ball_vel_mm_s if self.use_kalman_derivative else None))
        elif isinstance(self.controller, LQRController):
            state = np.array([
                ball_pos_mm[0] / 1000.0, ball_pos_mm[1] / 1000.0,
                ball_vel_mm_s[0] / 1000.0, ball_vel_mm_s[1] / 1000.0
            ])
            target = np.array([
                target_pos_mm[0] / 1000.0, target_pos_mm[1] / 1000.0
            ])

            rx, ry = self.controller.update(state, target, dt)
        else:
            return 0.0, 0.0

        rx, ry, clipped = clip_tilt_vector(rx, ry)
        return rx, ry

    def on_controller_param_change(self, param_name, value):
        """Handle controller parameter changes."""
        if self.controller is None:
            return

        if isinstance(self.controller, PIDController):
            if param_name == 'kp':
                self.controller.kp = value
            elif param_name == 'ki':
                self.controller.ki = value
            elif param_name == 'kd':
                self.controller.kd = value
            self.log(f"PID {param_name.upper()}: {value:.6f}")
        elif isinstance(self.controller, LQRController):
            if param_name == 'Q_pos':
                self.controller.Q_pos = value
            elif param_name == 'Q_vel':
                self.controller.Q_vel = value
            elif param_name == 'R':
                self.controller.R = value
            self.controller.recompute_lqr_gain()
            self.log(f"LQR {param_name}: {value:.2f}")

    # ============================================================================
    # HARDWARE COMMUNICATION
    # ============================================================================

    def connect_serial(self):
        """Connect to hardware via serial port."""
        port = self.port_var

        if not port:
            self.log("Error: No serial port selected")
            return

        if self.connected:
            self.log("Already connected")
            return

        self.serial_controller = SerialController(port)
        success, message = self.serial_controller.connect()

        if success:
            self.connected = True
            self.log(f"Connected to {port}")

            time.sleep(HardwareConnectionConfig.POST_CONNECTION_DELAY_S)
            self.serial_controller.set_servo_speed(0)
            time.sleep(HardwareConnectionConfig.POST_SERVO_SPEED_DELAY_S)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(HardwareConnectionConfig.POST_SERVO_ACCEL_DELAY_S)
            self.log("Servo parameters configured: Speed=0, Acceleration=0")

            success_timer, msg_timer = self.timer_manager.set_high_resolution()
            self.log(msg_timer)

            if hasattr(self, 'gui_modules') and 'serial_connection' in self.gui_modules:
                self.gui_modules['serial_connection'].update({
                    'connected': True,
                    'port': port
                })
        else:
            self.log(f"Connection failed: {message}")
            if hasattr(self, 'gui_modules') and 'serial_connection' in self.gui_modules:
                self.gui_modules['serial_connection'].update({'connected': False})

    def disconnect_serial(self):
        """Disconnect from hardware."""
        if not self.connected:
            return

        if self.serial_controller:
            self.serial_controller.disconnect()
            self.serial_controller = None

        self.timer_manager.restore_default()
        self.connected = False

        if hasattr(self, 'gui_modules') and 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection'].update({'connected': False})

        self.log("Disconnected")

    def initialize_ik_cache(self):
        """Pre-compute common inverse kinematics solutions."""
        if not hasattr(self, 'ik_cache') or self.ik_cache is None:
            self.ik_cache = IKCache(max_size=PerformanceConfig.IK_CACHE_SIZE)

        self.log("Initializing IK cache...")
        tilts = np.arange(PerformanceConfig.IK_PREWARM_TILT_RANGE[0],
                          PerformanceConfig.IK_PREWARM_TILT_RANGE[1] + 1,
                          PerformanceConfig.IK_PREWARM_TILT_STEP)
        count = 0
        for rx in tilts:
            for ry in tilts:
                translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                rotation = np.array([float(rx), float(ry), 0.0])
                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                if angles is not None:
                    self.ik_cache.put(translation, rotation, angles)
                    count += 1

        self.log(f"IK cache initialized with {count} pre-computed solutions")

    # ============================================================================
    # CONTROL THREAD
    # ============================================================================

    def start_simulation(self):
        """Start control loop (hardware or simulation mode)."""
        if self.operation_mode == 'real':
            if not self.connected:
                self.log("Connect to hardware first")
                return

            if self.simulation_running:
                return

            self.simulation_running = True
            self.simulation_time = 0.0

            # Update button states
            if 'simulation_control' in self.gui_modules:
                sim_ctrl = self.gui_modules['simulation_control']
                sim_ctrl.start_btn.setEnabled(False)
                sim_ctrl.stop_btn.setEnabled(True)

            gc.disable()
            self.log(f"Control started ({self.control_frequency}Hz, GC disabled)")

            self.control_thread = threading.Thread(target=self._control_thread_func, daemon=True)
            self.control_thread.start()

            # Set thread priority
            thread_id = self.control_thread.ident
            if thread_id:
                success = self.priority_manager.set_thread_priority(thread_id, priority=1)
                if success:
                    self.log("Control thread priority set to ABOVE_NORMAL")

            self.last_gui_update = time.time()
            self._gui_update_loop()
        else:
            # Call parent's simulation mode
            super().start_simulation()

    def _control_thread_func(self):
        """Hardware control thread - continuously read sensors and update servos."""
        self.log("Control thread started")

        while self.simulation_running:
            loop_start = time.perf_counter()

            # Get ball data from Pixy2 camera via serial
            ball_data = self.serial_controller.get_latest_ball_data()

            if ball_data is not None:
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                # Camera dimensions: 316×208 pixels, origin at top-left
                CAMERA_HEIGHT_PIXELS = 208.0

                ball_x_mm = (pixy_x - Pixy2CameraConfig.CENTER_X) * self.pixels_to_mm_x
                ball_y_mm = (CAMERA_HEIGHT_PIXELS - pixy_y - Pixy2CameraConfig.CENTER_Y) * self.pixels_to_mm_y

                self.ball_pos_mm = np.array([ball_x_mm, ball_y_mm])
                self.ball_detected = ball_data.get('detected', False)

                # Update ball_pos tensor for plotting (convert mm to m)
                self.ball_pos[0, 0] = ball_x_mm / 1000.0
                self.ball_pos[0, 1] = ball_y_mm / 1000.0

                # Track ball history for trail (only if detected)
                if self.ball_detected:
                    self.ball_history_x.append(ball_x_mm)
                    self.ball_history_y.append(ball_y_mm)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)

            # Kalman filter (only if enabled)
            if self.kalman_enabled:
                # Predict step (always predict, even without new measurement)
                rx_deg = self.prev_platform_angles.get('rx', 0.0)
                ry_deg = self.prev_platform_angles.get('ry', 0.0)
                self.kalman_filter.predict([rx_deg, ry_deg])

                # Update step (only if we have a recent ball measurement)
                if ball_data is not None and self.ball_detected:
                    self.kalman_filter.update(self.ball_pos_mm, self.simulation_time)

                # Extract Kalman state
                ball_pos_filtered = self.kalman_filter.get_position_mm()
                ball_vel_filtered = self.kalman_filter.get_velocity_mm_s()
            else:
                ball_pos_filtered = self.ball_pos_mm.copy()
                ball_vel_filtered = np.array([0.0, 0.0])

            # Controller update (if enabled)
            if self.controller_enabled and self.controller is not None:
                target_pos_mm = np.array([self.target_x * 1000.0, self.target_y * 1000.0])

                ik_start = time.perf_counter()
                rx_ctrl, ry_ctrl = self._update_controller(
                    ball_pos_filtered,
                    ball_vel_filtered,
                    target_pos_mm,
                    self.control_interval
                )
                ik_time = time.perf_counter() - ik_start

                # Use controller output directly (no IMU in base class)
                rx = rx_ctrl
                ry = ry_ctrl

                # Update dof_values so GUI reflects current state
                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

                # Calculate servo angles
                translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                rotation = np.array([rx, ry, 0.0])

                # Check cache first
                if self.ik_cache:
                    angles = self.ik_cache.get(translation, rotation)
                    if angles is None:
                        angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)
                        if angles is not None:
                            self.ik_cache.put(translation, rotation, angles)
                else:
                    angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)

                if angles is not None:
                    serial_start = time.perf_counter()
                    self.serial_controller.send_servo_angles(angles)
                    serial_time = time.perf_counter() - serial_start

                    self.last_cmd_angles = angles

                    self.prev_platform_angles['rx'] = rx
                    self.prev_platform_angles['ry'] = ry

                    # Track performance
                    if len(self.performance_data['ik_times']) < 1000:
                        self.performance_data['ik_times'].append(ik_time)
                        self.performance_data['serial_times'].append(serial_time)

            # Manual control when controller disabled
            elif not self.controller_enabled:
                rx = self.dof_values['rx']
                ry = self.dof_values['ry']

                translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
                rotation = np.array([rx, ry, self.dof_values['rz']])

                angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset)

                if angles is not None:
                    self.serial_controller.send_servo_angles(angles)
                    self.last_cmd_angles = angles

            # Track loop time
            loop_time = time.perf_counter() - loop_start
            if len(self.performance_data['loop_times']) < 1000:
                self.performance_data['loop_times'].append(loop_time)

            # Sleep to maintain control frequency
            sleep_time = max(0.0, self.control_interval - loop_time)
            time.sleep(sleep_time)

            self.simulation_time += self.control_interval

        self.log("Control thread stopped")

    def _gui_update_loop(self):
        """Update GUI at lower rate than control loop."""
        if not self.simulation_running:
            return

        # Update GUI modules
        self.update_gui_modules()

        # Update plots
        if hasattr(self, 'ball_trail') and self.ball_history_x and len(self.ball_history_x) > 1:
            self.ball_trail.setData(self.ball_history_x, self.ball_history_y)

        # Schedule next update
        QTimer.singleShot(GUIConfig.PLOT_DISABLED_INTERVAL_MS, self._gui_update_loop)

    def update_gui_modules(self):
        """Update GUI modules with current state."""
        # In simulation mode, use parent's comprehensive update
        if self.operation_mode == 'sim':
            super().update_gui_modules()
            return

        # Hardware mode: partial updates (hardware-specific)
        if not hasattr(self, 'gui_modules'):
            return

        # Update simulation control (button states and time)
        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].update({
                'simulation_time': self.simulation_time,
                'simulation_running': self.simulation_running
            })

        # Update ball state
        if 'ball_state' in self.gui_modules:
            ball_pos_filtered = self.kalman_filter.get_position_mm() if self.kalman_enabled else self.ball_pos_mm
            ball_vel_filtered = self.kalman_filter.get_velocity_mm_s() if self.kalman_enabled else np.array([0.0, 0.0])

            self.gui_modules['ball_state'].update({
                'pos_x': ball_pos_filtered[0],
                'pos_y': ball_pos_filtered[1],
                'vel_x': ball_vel_filtered[0],
                'vel_y': ball_vel_filtered[1],
                'detected': self.ball_detected,
                'time_since_update': self.simulation_time - self.last_ball_update
            })

        # Update platform state (calculate FK from servo angles)
        if 'platform_state' in self.gui_modules and hasattr(self, 'last_cmd_angles'):
            translation_fk, rotation_fk, fk_success, _ = self.ik.calculate_forward_kinematics(
                self.last_cmd_angles,
                use_top_surface_offset=self.use_top_surface_offset
            )

            if fk_success:
                self.gui_modules['platform_state'].update({
                    'x': translation_fk[0],
                    'y': translation_fk[1],
                    'z': translation_fk[2],
                    'rx': rotation_fk[0],
                    'ry': rotation_fk[1],
                    'rz': rotation_fk[2]
                })

        # Update controller output
        if 'controller_output' in self.gui_modules:
            self.gui_modules['controller_output'].update({
                'rx': self.prev_platform_angles['rx'],
                'ry': self.prev_platform_angles['ry']
            })

    # ============================================================================
    # MODE AND CONTROLLER SWITCHING
    # ============================================================================

    def on_mode_change(self, mode):
        """Handle mode change (simulation ↔ real)."""
        if self.simulation_running:
            self.log("Stop simulation before changing mode")
            return

        self.operation_mode = mode
        self.log(f"Mode changed to: {mode.upper()}")

        # Disconnect hardware if switching to simulation
        if mode == 'sim' and self.connected:
            self.disconnect_serial()

        # Rebuild GUI with new controller config
        self._rebuild_gui()

    def on_controller_type_change(self, controller_type):
        """Handle controller type change (PID ↔ LQR ↔ Manual)."""
        if self.simulation_running:
            self.log("Stop simulation before changing controller")
            return

        self.controller_type_selection = controller_type
        self.log(f"Controller type changed to: {controller_type}")

        # Rebuild GUI
        self._rebuild_gui()

    def _rebuild_gui(self):
        """Rebuild GUI after mode/controller change."""
        # Stop and clear current GUI
        if self.simulation_running:
            self.stop_simulation()

        # Update controller config
        self.controller_config = self._create_controller_config()

        # Clear old GUI modules
        if hasattr(self, 'gui_modules'):
            for module in self.gui_modules.values():
                if module and hasattr(module, 'widget'):
                    try:
                        module.widget.deleteLater()
                    except:
                        pass

        # Replace central widget
        old_central = self.central_widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        old_central.deleteLater()

        gc.collect()

        # Rebuild GUI
        self._build_modular_gui()

        QApplication.processEvents()

        # Reinitialize controller
        self._initialize_controller()

        self.log("GUI rebuilt")
