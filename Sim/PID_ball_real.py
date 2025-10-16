#!/usr/bin/env python3
"""
Stewart Platform Real Hardware Controller - PHASE 4 COMPLETE

Features:
- 100Hz dedicated control thread
- Pixy2 camera integration
- Modular GUI with scrollable columns
- Phase 1: Ball position EMA filtering
- Phase 2: Garbage collection optimization
- Phase 3: Optimized baud rates (USB 200k, Maestro 250k)
- Phase 4: Windows thread priority
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

from base_simulator import BaseStewartSimulator
from hardware_controller_config import HardwareControllerConfig, SerialController, IKCache
from control_core import BallPositionFilter
from utils import ControlLoopConfig, GUIConfig, MAX_SERVO_ANGLE_DEG, format_time, format_vector_2d
from gui_builder import create_standard_layout

# ============================================================================
# PHASE 4: WINDOWS THREAD PRIORITY MANAGER
# ============================================================================

# Windows thread priority constants
THREAD_PRIORITY_IDLE = -15
THREAD_PRIORITY_LOWEST = -2
THREAD_PRIORITY_BELOW_NORMAL = -1
THREAD_PRIORITY_NORMAL = 0
THREAD_PRIORITY_ABOVE_NORMAL = 1
THREAD_PRIORITY_HIGHEST = 2
THREAD_PRIORITY_TIME_CRITICAL = 15


class ThreadPriorityManager:
    """
    Windows thread priority manager.

    Elevates control thread priority to reduce context switches and jitter.
    No-op on Linux/Mac.
    """

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
            priority: 1=ABOVE_NORMAL, 2=HIGHEST

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
        except Exception as e:
            return False


# ============================================================================
# HARDWARE CONTROLLER
# ============================================================================

class HardwareStewartSimulator(BaseStewartSimulator):
    """Hardware-specific Stewart Platform Simulator with modular GUI."""

    def __init__(self, root):
        self.port_var = tk.StringVar()
        config = HardwareControllerConfig()

        # Initialize ball filter BEFORE calling super().__init__()
        self.ball_filter = BallPositionFilter(alpha=0.3)

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

        self.control_thread = None
        self.last_sent_angles = None
        self.angle_change_threshold = 0.2

        # Phase 4: Thread priority manager
        self.priority_manager = ThreadPriorityManager()
        self.control_thread_id = None

        self.actual_fps = 0.0
        self.timing_stats = {
            'ik_time': [],
            'send_time': [],
            'total_time': []
        }
        self.ik_timeout_count = 0

        self.last_gui_update = time.time()
        self.gui_update_count = 0

        # Disable start button until connected
        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.config(state='disabled')

        self.log("Hardware controller initialized (100Hz mode)")
        self.log("Phase 1: Ball EMA filter enabled")
        self.log("Phase 2: GC optimization enabled")
        self.log("Phase 3: Baud rates optimized (USB 200k, Maestro 250k)")
        self.log("Phase 4: Thread priority manager ready (Windows only)")

    def _create_controller_param_widgets(self):
        """Override to use hardware-specific defaults."""
        self.param_definitions = [
            ('kp', 'P (Proportional)', 3.0, 3),
            ('ki', 'I (Integral)', 1.0, 3),
            ('kd', 'D (Derivative)', 3.0, 3)
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

        layout = create_standard_layout(
            scrollable_columns=True,
            include_plot=True
        )

        # Column 1 (400px, scrollable): Hardware controls and configuration
        layout['columns'][0]['modules'] = [
            {'type': 'performance_stats'},
            {'type': 'ball_filter',
             'args': {'ball_filter': self.ball_filter}},
            {'type': 'serial_connection',
             'args': {'port_var': self.port_var}},
            {'type': 'simulation_control'},
            {'type': 'controller',
             'args': {'controller_config': self.controller_config,
                      'controller_widgets': self.controller_widgets}},
            {'type': 'trajectory_pattern',
             'args': {'pattern_var': self.pattern_type}},
            {'type': 'ball_state'},
            {'type': 'configuration',
             'args': {'use_offset_var': self.use_top_surface_offset}},
        ]

        # Column 2 (450px, scrollable): Status displays, manual control, and log
        layout['columns'][1]['modules'] = [
            {'type': 'servo_angles',
             'args': {'show_actual': False}},
            {'type': 'platform_pose'},
            {'type': 'controller_output',
             'args': {'controller_name': 'PID (Hardware)'}},
            {'type': 'manual_pose',
             'args': {'dof_config': self.dof_config}},
            {'type': 'debug_log',
             'args': {'height': 8}},
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

            self.prewarm_ik_cache()

            # Enable start button
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

        # Disable start button
        if 'simulation_control' in self.gui_modules:
            self.gui_modules['simulation_control'].start_btn.config(state='disabled')

        # Reset filter
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

        # Phase 2: Disable garbage collection during control loop
        gc.disable()
        self.log("Control started (100Hz, GC disabled)")

        self.control_thread = threading.Thread(target=self._control_thread_func,
                                               daemon=True)
        self.control_thread.start()

        # Phase 4: Set control thread to high priority (Windows only)
        self.control_thread_id = self.control_thread.ident
        if self.priority_manager.set_thread_priority(self.control_thread_id, THREAD_PRIORITY_ABOVE_NORMAL):
            self.log("Thread priority: ABOVE_NORMAL (Phase 4)")
        else:
            if sys.platform.startswith('win'):
                self.log("Note: Could not set thread priority (admin may be required)")

        self.last_gui_update = time.time()
        self.gui_update_count = 0
        self._gui_update_loop()

    def _control_thread_func(self):
        """Dedicated 100Hz control thread."""
        loop_interval = ControlLoopConfig.INTERVAL_S
        max_ik_time = ControlLoopConfig.IK_TIMEOUT_S

        while self.simulation_running:
            loop_start = time.perf_counter()

            ball_data = self.serial_controller.get_latest_ball_data()

            if ball_data is not None:
                self.last_ball_update = self.simulation_time

                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                ball_x_mm = (pixy_x - 158.0) * self.pixels_to_mm_x
                ball_y_mm = -(pixy_y - 104.0) * self.pixels_to_mm_y

                # Phase 1: Apply EMA filter to smooth camera noise
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

            if self.controller_enabled.get() and self.ball_detected:
                pattern_time = self.simulation_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                target_pos_mm = (target_x, target_y)

                rx, ry = self.controller.update(self.ball_pos_mm, target_pos_mm,
                                                loop_interval)

                ry = -ry  # Camera Y-axis inversion

                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

                start_ik = time.perf_counter()

                translation = np.array([
                    self.dof_values['x'],
                    self.dof_values['y'],
                    self.dof_values['z']
                ])
                rotation = np.array([
                    self.dof_values['rx'],
                    self.dof_values['ry'],
                    self.dof_values['rz']
                ])

                angles = self.ik_cache.get(translation, rotation)

                if angles is None:
                    angles = self.ik.calculate_servo_angles(
                        translation, rotation,
                        self.use_top_surface_offset.get()
                    )

                    ik_time = time.perf_counter() - start_ik

                    if ik_time > max_ik_time:
                        if self.last_sent_angles is not None:
                            angles = self.last_sent_angles
                            self.ik_timeout_count += 1
                    elif angles is not None:
                        self.ik_cache.put(translation, rotation, angles)
                        self.timing_stats['ik_time'].append(ik_time * 1000)
                else:
                    ik_time = time.perf_counter() - start_ik
                    self.timing_stats['ik_time'].append(ik_time * 1000)

                if angles is not None:
                    if (self.last_sent_angles is None or
                            not np.allclose(angles, self.last_sent_angles,
                                            atol=self.angle_change_threshold)):

                        send_start = time.perf_counter()
                        success = self.serial_controller.send_servo_angles(angles)
                        send_time = (time.perf_counter() - send_start) * 1000

                        if success:
                            self.last_sent_angles = angles.copy()

                            total_time = (time.perf_counter() - loop_start) * 1000
                            self.timing_stats['send_time'].append(send_time)
                            self.timing_stats['total_time'].append(total_time)

                            for key in self.timing_stats:
                                if len(self.timing_stats[key]) > 1000:
                                    self.timing_stats[key].pop(0)

            self.simulation_time += loop_interval

            elapsed = time.perf_counter() - loop_start
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _gui_update_loop(self):
        """Separate GUI update loop at lower frequency."""
        if not self.simulation_running:
            return

        # Update GUI modules
        self.update_gui_modules()

        # Update plot
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

        # Controller output
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

        # Servo angles
        if self.last_sent_angles is not None:
            state['cmd_angles'] = self.last_sent_angles

        # Pattern info
        pattern_configs = {
            'static': "Tracking: Center (0, 0)",
            'circle': "Tracking: Circle (r=50mm, T=10s)",
            'figure8': "Tracking: Figure-8 (60×40mm, T=12s)",
            'star': "Tracking: 5-Point Star (r=60mm, T=15s)"
        }
        state['pattern_info'] = pattern_configs.get(self.pattern_type.get(), "")

        self.gui_builder.update_modules(state)

    def setup_plot(self):
        """Setup plot for hardware (override base class)."""
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
        """Show performance statistics."""
        stats_msg = "Performance Statistics (100Hz Hardware Mode - Phase 4)\n"
        stats_msg += "=" * 50 + "\n\n"

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
        stats_msg += f"  Phase 1: EMA Filter (α={self.ball_filter.get_alpha():.2f})\n"
        stats_msg += f"  Phase 2: GC Disabled during control\n"
        stats_msg += f"  Phase 3: USB 200k, Maestro 250k baud\n"
        stats_msg += f"  Phase 4: Thread Priority ABOVE_NORMAL\n"
        stats_msg += f"  IK Timeouts: {self.ik_timeout_count}\n"

        messagebox.showinfo("Performance Statistics", stats_msg)

    def calculate_ik(self):
        """Calculate inverse kinematics and send to hardware."""
        translation = np.array([self.dof_values['x'],
                                self.dof_values['y'],
                                self.dof_values['z']])

        from control_core import clip_tilt_vector
        from utils import MAX_TILT_ANGLE_DEG

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

        # Phase 2: Re-enable GC and clean up
        gc.enable()
        gc.collect()

        self.log("Control stopped (GC re-enabled)")

    def on_closing(self):
        """Clean shutdown."""
        if self.simulation_running:
            self.stop_simulation()

        if self.connected:
            self.disconnect_serial()

        # Ensure GC is re-enabled on exit
        gc.enable()
        gc.collect()

        super().on_closing()


def main():
    """Launch hardware controller."""
    root = tk.Tk()
    app = HardwareStewartSimulator(root)

    app.log("=" * 50)
    app.log("Hardware Controller - Phase 4 Complete")
    app.log("=" * 50)
    app.log("")
    app.log("Optimizations Active:")
    app.log("  ✓ Phase 1: EMA Ball Filter")
    app.log("  ✓ Phase 2: GC Optimization")
    app.log("  ✓ Phase 3: Optimized Baud Rates")
    app.log("  ✓ Phase 4: Windows Thread Priority")
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