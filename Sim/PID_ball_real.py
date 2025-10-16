#!/usr/bin/env python3
"""
Stewart Platform Real Hardware Controller

Features:
- 100Hz dedicated control thread
- Pixy2 camera integration
- Modular GUI with scrollable columns
- Ball position EMA filtering
- Garbage collection optimization
- Optimized baud rates (USB 200k, Maestro 250k)
- Windows thread priority
- Windows timer resolution + Pre-allocated NumPy arrays
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
import time
import threading
import serial.tools.list_ports
import gc
import sys
import ctypes

from setup.base_simulator import BaseStewartSimulator
from setup.hardware_controller_config import HardwareControllerConfig, SerialController, IKCache
from core.control_core import BallPositionFilter
from core.utils import ControlLoopConfig, GUIConfig, MAX_SERVO_ANGLE_DEG, format_time, format_vector_2d
from gui.gui_builder import create_standard_layout


THREAD_PRIORITY_IDLE = -15
THREAD_PRIORITY_LOWEST = -2
THREAD_PRIORITY_BELOW_NORMAL = -1
THREAD_PRIORITY_NORMAL = 0
THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_HIGHEST = 2
THREAD_PRIORITY_TIME_CRITICAL = 15


class WindowsTimerManager:
    """Windows multimedia timer resolution manager. Reduces timer granularity from 15.6ms to 1ms."""

    def __init__(self):
        self.timer_set = False
        self.is_windows = sys.platform.startswith('win')

    def set_high_resolution(self):
        """Set Windows timer to 1ms resolution."""
        if not self.is_windows:
            return False, "Not Windows - timer not set"

        try:
            timeBeginPeriod = ctypes.windll.winmm.timeBeginPeriod
            result = timeBeginPeriod(1)
            if result == 0:
                self.timer_set = True
                return True, "Windows timer set to 1ms"
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
    """Windows thread priority manager. Elevates control thread priority to reduce jitter."""

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


class HardwareStewartSimulator(BaseStewartSimulator):
    """Hardware-specific Stewart Platform Simulator with modular GUI."""

    def __init__(self, root):
        self.port_var = tk.StringVar()
        config = HardwareControllerConfig()

        self.ball_filter = BallPositionFilter(alpha=0.7)

        super().__init__(root, config)

        self.root.title("Stewart Platform - Real Hardware Control (100Hz)")

        self.serial_controller = None
        self.connected = False

        self.pixy_width_mm = 350.0
        self.pixy_height_mm = 266.0
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0

        self.ball_pos_mm = (0.0, 0.0)
        self.ball_detected = False
        self.last_ball_update = 0
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = 100

        self.ik_cache = IKCache(max_size=5000)

        self._translation_buffer = np.zeros(3, dtype=np.float64)
        self._rotation_buffer = np.zeros(3, dtype=np.float64)

        self.control_thread = None
        self.last_sent_angles = None
        self.angle_change_threshold = 0.2

        self.priority_manager = ThreadPriorityManager()
        self.control_thread_id = None

        self.timer_manager = WindowsTimerManager()

        self.actual_fps = 0.0
        self.timing_stats = {
            'ik_time': [],
            'send_time': [],
            'total_time': []
        }
        self.timing_breakpoints = {}
        self.ik_timeout_count = 0

        self.last_gui_update = time.time()
        self.gui_update_count = 0

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.config(state='disabled')

        self.log("Hardware controller initialized (100Hz mode)")
        self.log("Optimizations: EMA filter, GC optimization, optimized baud rates, thread priority, timer resolution")

    def _create_controller_param_widgets(self):
        """Override to use hardware-specific defaults."""
        self.param_definitions = [
            ('kp', 'P (Proportional)', 1.0, 6),
            ('ki', 'I (Integral)', 1.0, 6),
            ('kd', 'D (Derivative)', 4.0, 5)
        ]

        self.controller_widgets = {
            'sliders': {},
            'value_labels': {},
            'scalar_vars': {},
            'update_fn': lambda: None,
            'param_definitions': self.param_definitions
        }

    def get_layout_config(self):
        """Define hardware-specific GUI layout with scrollable columns."""
        layout = create_standard_layout(scrollable_columns=False, include_plot=True)

        layout['columns'][0]['modules'] = [
            {'type': 'performance_stats'},
            {'type': 'ball_filter', 'args': {'ball_filter': self.ball_filter}},
            {'type': 'serial_connection', 'args': {'port_var': self.port_var}},
            {'type': 'simulation_control'},
            {'type': 'controller',
             'args': {'controller_config': self.controller_config,
                      'controller_widgets': self.controller_widgets}},
            {'type': 'trajectory_pattern', 'args': {'pattern_var': self.pattern_type}},
            {'type': 'ball_state'},
            {'type': 'configuration', 'args': {'use_offset_var': self.use_top_surface_offset}},
        ]

        layout['columns'][1]['modules'] = [
            {'type': 'servo_angles', 'args': {'show_actual': False}},
            {'type': 'platform_pose'},
            {'type': 'controller_output', 'args': {'controller_name': 'PID (Hardware)'}},
            {'type': 'manual_pose', 'args': {'dof_config': self.dof_config}},
            {'type': 'debug_log', 'args': {'height': 8}},
        ]

        return layout

    def _create_callbacks(self):
        """Create callback dictionary including hardware-specific callbacks."""
        callbacks = super()._create_callbacks()

        callbacks.update({
            'connect': self.connect_serial,
            'disconnect': self.disconnect_serial,
            'show_stats': self.show_timing_stats,
        })

        return callbacks

    def refresh_ports(self):
        """Refresh available serial ports."""
        if 'serial_connection' in self.gui_modules:
            self.gui_modules['serial_connection']._refresh_ports()

    def prewarm_ik_cache(self):
        """Pre-calculate common IK solutions."""
        self.log("Pre-warming IK cache...")

        tilts = np.arange(-15, 16, 2)
        count = 0
        start_time = time.time()

        for rx in tilts:
            for ry in tilts:
                translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
                rotation = np.array([float(rx), float(ry), 0.0])

                angles = self.ik.calculate_servo_angles(
                    translation, rotation,
                    self.use_top_surface_offset.get()
                )

                if angles is not None:
                    self.ik_cache.put(translation, rotation, angles)
                    count += 1

        elapsed = time.time() - start_time
        self.log(f"Pre-warmed {count} poses in {elapsed:.2f}s")

    def connect_serial(self):
        """Connect to hardware."""
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Error", "No port selected")
            return

        self.serial_controller = SerialController(port)
        success, message = self.serial_controller.connect()

        if success:
            self.connected = True
            self.log(f"Connected to {port}")

            time.sleep(0.5)
            self.serial_controller.set_servo_speed(0)
            time.sleep(0.1)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(0.2)
            self.log("Servos: Speed=0 (unlimited), Accel=0")

            success_timer, msg_timer = self.timer_manager.set_high_resolution()
            self.log(msg_timer)

            self.prewarm_ik_cache()

            if 'simulation_control' in self.gui_modules:
                self.gui_modules['simulation_control'].start_btn.config(state='normal')
        else:
            messagebox.showerror("Error", message)
            self.log(f"Error: {message}")

    def disconnect_serial(self):
        """Disconnect from hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False

        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.config(state='disabled')

        self.ball_filter.reset()

        self.log("Disconnected")

    def _initialize_controller(self):
        """Initialize hardware PID controller."""
        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
        ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
        kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

        self.controller = self.controller_config.create_controller(
            kp=kp, ki=ki, kd=kd, output_limit=15.0
        )

        self.log(f"PID initialized: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

    def on_controller_param_change(self):
        """Update controller when parameters change."""
        if self.controller is None:
            return

        sliders = self.controller_widgets['sliders']
        scalar_vars = self.controller_widgets['scalar_vars']

        kp = self.controller_config.get_scaled_param('kp', sliders, scalar_vars)
        ki = self.controller_config.get_scaled_param('ki', sliders, scalar_vars)
        kd = self.controller_config.get_scaled_param('kd', sliders, scalar_vars)

        self.controller.set_gains(kp, ki, kd)

        if self.controller_enabled.get():
            self.log(f"PID gains updated: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

    def start_simulation(self):
        """Start 100Hz hardware control thread."""
        if not self.connected:
            return

        self.simulation_running = True
        self.simulation_time = 0.0
        self.ik_timeout_count = 0

        gc.disable()
        self.log("Control started (100Hz, GC disabled)")

        self.control_thread = threading.Thread(target=self._control_thread_func, daemon=True)
        self.control_thread.start()

        self.control_thread_id = self.control_thread.ident
        if self.priority_manager.set_thread_priority(self.control_thread_id, THREAD_PRIORITY_TIME_CRITICAL):
            self.log("Thread priority: TIME_CRITICAL")
        else:
            if sys.platform.startswith('win'):
                self.log("Note: Could not set thread priority")

        self.last_gui_update = time.time()
        self.gui_update_count = 0
        self._gui_update_loop()

    def _control_thread_func(self):
        """Dedicated 100Hz control thread with detailed timing instrumentation."""
        loop_interval = ControlLoopConfig.INTERVAL_S
        max_ik_time = ControlLoopConfig.IK_TIMEOUT_S

        timing_breakpoints = {
            'ball_read': [],
            'ball_process': [],
            'pattern_calc': [],
            'pid_update': [],
            'ik_total': [],
            'serial_send': [],
            'sleep': []
        }
        max_breakpoint_samples = 1000

        self.timing_breakpoints = timing_breakpoints

        while self.simulation_running:
            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            ball_data = self.serial_controller.get_latest_ball_data()
            ball_read_time = (time.perf_counter() - t0) * 1000
            timing_breakpoints['ball_read'].append(ball_read_time)

            if ball_data is not None:
                t1 = time.perf_counter()
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                ball_x_mm = (pixy_x - 158.0) * self.pixels_to_mm_x
                ball_y_mm = -(pixy_y - 104.0) * self.pixels_to_mm_y

                ball_x_mm_filtered, ball_y_mm_filtered = self.ball_filter.update(
                    ball_x_mm, ball_y_mm
                )
                self.ball_pos_mm = (ball_x_mm_filtered, ball_y_mm_filtered)
                self.ball_detected = ball_data['detected']

                if self.ball_detected:
                    self.ball_history_x.append(ball_x_mm_filtered)
                    self.ball_history_y.append(ball_y_mm_filtered)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)

                ball_process_time = (time.perf_counter() - t1) * 1000
                timing_breakpoints['ball_process'].append(ball_process_time)
            else:
                timing_breakpoints['ball_process'].append(0.0)

            if self.controller_enabled.get() and self.ball_detected:
                t2 = time.perf_counter()
                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                target_pos_mm = (target_x, target_y)
                pattern_calc_time = (time.perf_counter() - t2) * 1000
                timing_breakpoints['pattern_calc'].append(pattern_calc_time)

                t3 = time.perf_counter()
                rx, ry = self.controller.update(self.ball_pos_mm, target_pos_mm, loop_interval)
                ry = -ry
                pid_update_time = (time.perf_counter() - t3) * 1000
                timing_breakpoints['pid_update'].append(pid_update_time)

                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

                start_ik = time.perf_counter()

                self._translation_buffer[0] = self.dof_values['x']
                self._translation_buffer[1] = self.dof_values['y']
                self._translation_buffer[2] = self.dof_values['z']

                self._rotation_buffer[0] = self.dof_values['rx']
                self._rotation_buffer[1] = self.dof_values['ry']
                self._rotation_buffer[2] = self.dof_values['rz']

                angles = self.ik_cache.get(self._translation_buffer, self._rotation_buffer)

                if angles is None:
                    angles = self.ik.calculate_servo_angles(
                        self._translation_buffer,
                        self._rotation_buffer,
                        self.use_top_surface_offset.get()
                    )

                    ik_time = time.perf_counter() - start_ik

                    if ik_time > max_ik_time:
                        if self.last_sent_angles is not None:
                            angles = self.last_sent_angles
                            self.ik_timeout_count += 1
                    elif angles is not None:
                        self.ik_cache.put(
                            self._translation_buffer,
                            self._rotation_buffer,
                            angles
                        )
                        self.timing_stats['ik_time'].append(ik_time * 1000)
                else:
                    ik_time = time.perf_counter() - start_ik
                    self.timing_stats['ik_time'].append(ik_time * 1000)

                ik_total_time = (time.perf_counter() - start_ik) * 1000
                timing_breakpoints['ik_total'].append(ik_total_time)

                if angles is not None:
                    if (self.last_sent_angles is None or
                            not np.allclose(angles, self.last_sent_angles,
                                            atol=self.angle_change_threshold)):

                        send_start = time.perf_counter()
                        success = self.serial_controller.send_servo_angles(angles)
                        send_time = (time.perf_counter() - send_start) * 1000
                        timing_breakpoints['serial_send'].append(send_time)

                        if success:
                            self.last_sent_angles = angles.copy()

                            total_time = (time.perf_counter() - loop_start) * 1000
                            self.timing_stats['send_time'].append(send_time)
                            self.timing_stats['total_time'].append(total_time)

                            for key in self.timing_stats:
                                if len(self.timing_stats[key]) > 1000:
                                    self.timing_stats[key].pop(0)
                    else:
                        timing_breakpoints['serial_send'].append(0.0)
                else:
                    timing_breakpoints['serial_send'].append(0.0)
            else:
                timing_breakpoints['pattern_calc'].append(0.0)
                timing_breakpoints['pid_update'].append(0.0)
                timing_breakpoints['ik_total'].append(0.0)
                timing_breakpoints['serial_send'].append(0.0)

            self.simulation_time += loop_interval

            t_sleep = time.perf_counter()
            elapsed = time.perf_counter() - loop_start

            if elapsed > 0.050:
                self.log(f"WARNING: Loop took {elapsed * 1000:.1f}ms - Windows preemption detected")
                timing_breakpoints['sleep'].append(0.0)
            else:
                sleep_time = loop_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                sleep_actual_time = (time.perf_counter() - t_sleep) * 1000
                timing_breakpoints['sleep'].append(sleep_actual_time)

            for key in timing_breakpoints:
                if len(timing_breakpoints[key]) > max_breakpoint_samples:
                    timing_breakpoints[key].pop(0)

    def _gui_update_loop(self):
        """Separate GUI update loop at lower frequency."""
        if not self.simulation_running:
            return

        self.update_gui_modules()

        if self.gui_update_count % 2 == 0:
            self._update_hardware_plot()

        self.gui_update_count += 1

        self.root.after(GUIConfig.UPDATE_INTERVAL_MS, self._gui_update_loop)

    def update_gui_modules(self):
        """Override to add hardware-specific state."""
        status = "Detected" if self.ball_detected else "Not detected"

        state = {
            'simulation_time': self.simulation_time,
            'controller_enabled': self.controller_enabled.get(),
            'ball_pos': self.ball_pos_mm,
            'ball_vel': status,
            'dof_values': self.dof_values,
            'connected': self.connected,
            'fps': ControlLoopConfig.FREQUENCY_HZ,
            'cache_hit_rate': self.ik_cache.get_hit_rate(),
            'ik_timeouts': self.ik_timeout_count,
        }

        if self.controller_enabled.get():
            rx = self.dof_values['rx']
            ry = self.dof_values['ry']
            magnitude = np.sqrt(rx ** 2 + ry ** 2)
            magnitude_percent = (magnitude / 15.0) * 100

            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)
            error_x = target_x - self.ball_pos_mm[0]
            error_y = target_y - self.ball_pos_mm[1]

            state['controller_output'] = (rx, ry)
            state['controller_magnitude'] = (magnitude, magnitude_percent)
            state['controller_error'] = (error_x, error_y)

        if self.last_sent_angles is not None:
            state['cmd_angles'] = self.last_sent_angles

        pattern_configs = {
            'static': "Tracking: Center (0, 0)",
            'circle': "Tracking: Circle (r=50mm, T=10s)",
            'figure8': "Tracking: Figure-8 (60×40mm, T=12s)",
            'star': "Tracking: 5-Point Star (r=60mm, T=15s)"
        }
        state['pattern_info'] = pattern_configs.get(self.pattern_type.get(), "")

        self.gui_builder.update_modules(state)

    def setup_plot(self):
        """Setup plot for hardware."""
        super().setup_plot()

        self.ball_trail, = self.ax.plot([], [], 'r-', alpha=0.3, linewidth=1,
                                        label='Ball Trail')

        legend = self.ax.legend(loc='upper right', fontsize=8,
                                facecolor=self.colors['panel_bg'],
                                edgecolor=self.colors['border'],
                                labelcolor=self.colors['fg'])
        legend.get_frame().set_alpha(0.9)

        self.canvas.draw()

    def _update_hardware_plot(self):
        """Update plot with hardware data."""
        try:
            if self.ball_detected:
                self.ball_circle.center = self.ball_pos_mm
                self.ball_circle.set_alpha(0.8)
            else:
                self.ball_circle.set_alpha(0.2)

            if len(self.ball_history_x) > 1:
                self.ball_trail.set_data(self.ball_history_x, self.ball_history_y)

            if self.pattern_type.get() != 'static':
                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                self.target_marker.set_data([target_x], [target_y])

            if self.tilt_arrow is not None:
                self.tilt_arrow.remove()
                self.tilt_arrow = None

            rx = self.dof_values['rx']
            ry = self.dof_values['ry']

            if abs(rx) > 0.5 or abs(ry) > 0.5:
                dx = -np.sin(np.radians(ry))
                dy = -np.sin(np.radians(rx))
                magnitude = np.sqrt(dx ** 2 + dy ** 2)

                if magnitude > 0:
                    dx = (dx / magnitude) * 30
                    dy = (dy / magnitude) * 30
                    color = self.colors['success']
                    self.tilt_arrow = self.ax.arrow(0, 0, dx, dy,
                                                    head_width=8, head_length=10,
                                                    fc=color, ec=color,
                                                    alpha=0.6, linewidth=2, zorder=5)

            self.canvas.draw_idle()
        except:
            pass

    def show_timing_stats(self):
        """Show performance statistics with detailed breakpoint analysis."""
        print("\n" + "=" * 70)
        print("DETAILED TIMING BREAKDOWN")
        print("=" * 70 + "\n")

        if hasattr(self, 'timing_breakpoints') and self.timing_breakpoints:
            breakpoint_names = {
                'ball_read': 'Ball Data Read (Queue)',
                'ball_process': 'Ball Processing (Filter/History)',
                'pattern_calc': 'Pattern Calculation',
                'pid_update': 'PID Controller Update',
                'ik_total': 'IK Total (Cache+Calc)',
                'serial_send': 'Serial Send',
                'sleep': 'Sleep/Timing'
            }

            for key, name in breakpoint_names.items():
                if key in self.timing_breakpoints and self.timing_breakpoints[key]:
                    data = [x for x in self.timing_breakpoints[key] if x > 0]
                    if data:
                        avg = np.mean(data)
                        max_val = np.max(data)
                        min_val = np.min(data)

                        marker = "SPIKE SOURCE!" if max_val > 50 else ""

                        print(f"{name}:{marker}")
                        print(f"  Average: {avg:.3f} ms")
                        print(f"  Min: {min_val:.3f} ms")
                        print(f"  Max: {max_val:.3f} ms")

                        if max_val > 10:
                            p95 = np.percentile(data, 95)
                            p99 = np.percentile(data, 99)
                            print(f"  95th percentile: {p95:.3f} ms")
                            print(f"  99th percentile: {p99:.3f} ms")
                        print()
        else:
            print("No timing breakpoint data collected yet!")

        print("=" * 70 + "\n")

        stats_msg = "Performance Statistics (100Hz Hardware Mode)\n"
        stats_msg += "=" * 60 + "\n\n"

        if self.timing_stats['ik_time']:
            stats_msg += "IK Calculation Time (ms):\n"
            stats_msg += f"  Average: {np.mean(self.timing_stats['ik_time']):.3f}\n"
            stats_msg += f"  Min: {np.min(self.timing_stats['ik_time']):.3f}\n"
            stats_msg += f"  Max: {np.max(self.timing_stats['ik_time']):.3f}\n\n"

            stats_msg += "Serial Send Time (ms):\n"
            stats_msg += f"  Average: {np.mean(self.timing_stats['send_time']):.3f}\n"
            stats_msg += f"  Min: {np.min(self.timing_stats['send_time']):.3f}\n"
            stats_msg += f"  Max: {np.max(self.timing_stats['send_time']):.3f}\n\n"

            stats_msg += "Total Loop Time (ms):\n"
            stats_msg += f"  Average: {np.mean(self.timing_stats['total_time']):.3f}\n"
            stats_msg += f"  Min: {np.min(self.timing_stats['total_time']):.3f}\n"
            stats_msg += f"  Max: {np.max(self.timing_stats['total_time']):.3f}\n\n"

        hit_rate = self.ik_cache.get_hit_rate()
        stats_msg += "IK Cache Statistics:\n"
        stats_msg += f"  Hit Rate: {hit_rate * 100:.1f}%\n"
        stats_msg += f"  Hits: {self.ik_cache.hits}\n"
        stats_msg += f"  Misses: {self.ik_cache.misses}\n"
        stats_msg += f"  Cache Size: {len(self.ik_cache.cache)}/{self.ik_cache.max_size}\n\n"

        stats_msg += "Optimizations Active:\n"
        stats_msg += f"  EMA Filter (α={self.ball_filter.get_alpha():.2f})\n"
        stats_msg += f"  GC Disabled during control\n"
        stats_msg += f"  USB 200k, Maestro 250k baud\n"
        stats_msg += f"  Thread Priority TIME_CRITICAL\n"
        stats_msg += f"  Windows Timer 1ms + Pre-allocated buffers\n"
        stats_msg += f"  IK Timeouts: {self.ik_timeout_count}\n\n"

        stats_msg += "DETAILED BREAKDOWN PRINTED TO CONSOLE"

        messagebox.showinfo("Performance Statistics", stats_msg)

    def calculate_ik(self):
        """Calculate inverse kinematics and send to hardware."""
        translation = np.array([self.dof_values['x'],
                                self.dof_values['y'],
                                self.dof_values['z']])

        from Sim.core.control_core import clip_tilt_vector
        from Sim.core.utils import MAX_TILT_ANGLE_DEG

        rx_limited, ry_limited, tilt_mag = clip_tilt_vector(
            self.dof_values['rx'],
            self.dof_values['ry'],
            MAX_TILT_ANGLE_DEG
        )

        if tilt_mag > MAX_TILT_ANGLE_DEG and not self.controller_enabled.get():
            self.dof_values['rx'] = rx_limited
            self.dof_values['ry'] = ry_limited

        rotation = np.array([rx_limited, ry_limited, self.dof_values['rz']])

        angles = self.ik.calculate_servo_angles(translation, rotation,
                                                self.use_top_surface_offset.get())

        if angles is not None:
            self.last_cmd_angles = angles

            if self.connected and not self.simulation_running:
                self.serial_controller.send_servo_angles(angles)

    def on_controller_toggle(self):
        """Override to handle manual control disabling for hardware."""
        enabled = self.controller_enabled.get()

        if enabled:
            self.controller.reset()
            self.ball_filter.reset()
            self.reset_pattern()
            self.log("PID control ENABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].config(state='disabled')
                manual_pose.sliders['ry'].config(state='disabled')
                manual_pose.sliders['x'].config(state='disabled')
                manual_pose.sliders['y'].config(state='disabled')
                manual_pose.sliders['z'].config(state='disabled')
        else:
            self.log("PID control DISABLED")

            if 'manual_pose' in self.gui_modules:
                manual_pose = self.gui_modules['manual_pose']
                manual_pose.sliders['rx'].config(state='normal')
                manual_pose.sliders['ry'].config(state='normal')
                manual_pose.sliders['x'].config(state='normal')
                manual_pose.sliders['y'].config(state='normal')
                manual_pose.sliders['z'].config(state='normal')

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Hardware controller update (not used - control thread handles it)."""
        return self.controller.update(ball_pos_mm, target_pos_mm, dt)

    def stop_simulation(self):
        """Stop the control thread."""
        self.simulation_running = False

        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        if self.serial_controller:
            while not self.serial_controller.command_queue.empty():
                try:
                    self.serial_controller.command_queue.get_nowait()
                except:
                    break

        gc.enable()
        gc.collect()

        self.log("Control stopped (GC re-enabled)")

    def on_closing(self):
        """Clean shutdown."""
        if self.simulation_running:
            self.stop_simulation()

        if self.connected:
            self.disconnect_serial()

        self.timer_manager.restore_default()

        gc.enable()
        gc.collect()

        super().on_closing()


def main():
    """Launch hardware controller."""
    root = tk.Tk()
    app = HardwareStewartSimulator(root)

    app.log("=" * 50)
    app.log("Hardware Controller - Ready")
    app.log("=" * 50)
    app.log("")
    app.log("Optimizations Active:")
    app.log("   EMA Ball Filter")
    app.log("   GC Optimization")
    app.log("   Optimized Baud Rates")
    app.log("   Windows Thread Priority")
    app.log("   Windows Timer + Pre-allocated Arrays")
    app.log("")
    app.log("Quick Start:")
    app.log("1. Select serial port and click 'Connect'")
    app.log("2. Tune ball filter with EMA slider")
    app.log("3. Enable PID Control for automatic balancing")
    app.log("4. Click 'Start' to begin 100Hz control loop")
    app.log("5. Select trajectory patterns to track")
    app.log("")

    root.mainloop()


if __name__ == "__main__":
    main()