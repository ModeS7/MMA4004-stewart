#!/usr/bin/env python3
"""
Stewart Platform Real Hardware Controller

Features:
- 100Hz dedicated control thread
- Pixy2 camera integration
- Maestro servo control
- IK caching for performance
- Vector-based tilt limiting
- Camera Y-axis inversion handling
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import time
import threading
import serial.tools.list_ports

from base_simulator import BaseStewartSimulator
from hardware_controller_config import HardwareControllerConfig, SerialController, IKCache
from utils import (
    ControlLoopConfig, GUIConfig, MAX_SERVO_ANGLE_DEG,
    format_time, format_vector_2d
)


class HardwareStewartSimulator(BaseStewartSimulator):
    """Hardware-specific Stewart Platform Simulator."""

    def __init__(self, root):
        config = HardwareControllerConfig()
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

        self.actual_fps = 0.0
        self.timing_stats = {
            'ik_time': [],
            'send_time': [],
            'total_time': []
        }
        self.ik_timeout_count = 0

        self.last_gui_update = time.time()
        self.gui_update_count = 0

        self._add_hardware_widgets()

        self.log("⚡ Hardware controller initialized (100Hz mode)")
        self.log("IK cache: 5000 entries (1mm/1° resolution)")

    def _add_hardware_widgets(self):
        """Add hardware-specific GUI widgets."""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                main_frame = widget
                break

        left_panel = None
        for child in main_frame.winfo_children():
            if isinstance(child, ttk.Frame):
                left_panel = child
                break

        if left_panel is None:
            return

        perf_frame = ttk.LabelFrame(left_panel, text="⚡ 100Hz MODE", padding=10)
        perf_frame.pack(fill='x', pady=(0, 10), before=left_panel.winfo_children()[0])

        self.fps_label = ttk.Label(perf_frame, text="Control Loop: 0 Hz",
                                   font=('Consolas', 10, 'bold'),
                                   foreground=self.colors['success'])
        self.fps_label.pack()

        self.cache_label = ttk.Label(perf_frame, text="IK Cache: 0.0%",
                                     font=('Consolas', 9))
        self.cache_label.pack()

        self.timeout_label = ttk.Label(perf_frame, text="IK Timeouts: 0",
                                       font=('Consolas', 9))
        self.timeout_label.pack()

        ttk.Button(perf_frame, text="Show Statistics",
                   command=self.show_timing_stats).pack(pady=(5, 0))

        conn_frame = ttk.LabelFrame(left_panel, text="Serial Connection", padding=10)
        conn_frame.pack(fill='x', pady=(0, 10), before=left_panel.winfo_children()[1])

        port_frame = ttk.Frame(conn_frame)
        port_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(port_frame, text="Port:").pack(side='left', padx=(0, 5))

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var,
                                       width=15, state='readonly')
        self.port_combo.pack(side='left', padx=5)
        self.refresh_ports()

        ttk.Button(port_frame, text="↻", command=self.refresh_ports,
                   width=3).pack(side='left')

        conn_btn_frame = ttk.Frame(conn_frame)
        conn_btn_frame.pack(fill='x', pady=(5, 0))

        self.connect_btn = ttk.Button(conn_btn_frame, text="Connect",
                                      command=self.connect_serial, width=12)
        self.connect_btn.pack(side='left', padx=5)

        self.disconnect_btn = ttk.Button(conn_btn_frame, text="Disconnect",
                                         command=self.disconnect_serial,
                                         state='disabled', width=12)
        self.disconnect_btn.pack(side='left', padx=5)

        self.conn_status_label = ttk.Label(conn_frame, text="Not connected",
                                           foreground=self.colors['border'])
        self.conn_status_label.pack(pady=(5, 0))

        self.start_btn.config(state='disabled')

    def refresh_ports(self):
        """Refresh available serial ports."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

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
        self.log(f"✓ Pre-warmed {count} poses in {elapsed:.2f}s")

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
            self.conn_status_label.config(text="Connected",
                                          foreground=self.colors['success'])
            self.connect_btn.config(state='disabled')
            self.disconnect_btn.config(state='normal')
            self.start_btn.config(state='normal')
            self.log(f"✓ Connected to {port}")

            time.sleep(0.5)
            self.serial_controller.set_servo_speed(0)
            time.sleep(0.1)
            self.serial_controller.set_servo_acceleration(0)
            time.sleep(0.2)
            self.log("✓ Servos: Speed=0 (unlimited), Accel=0")

            self.prewarm_ik_cache()
        else:
            messagebox.showerror("Error", message)
            self.log(f"✗ {message}")

    def disconnect_serial(self):
        """Disconnect from hardware."""
        if self.simulation_running:
            self.stop_simulation()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False
        self.conn_status_label.config(text="Not connected",
                                      foreground=self.colors['border'])
        self.connect_btn.config(state='normal')
        self.disconnect_btn.config(state='disabled')
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
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

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.log("▶ Control started (100Hz dedicated thread)")

        self.control_thread = threading.Thread(target=self._control_thread_func,
                                               daemon=True)
        self.control_thread.start()

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

                self.ball_pos_mm = (ball_x_mm, ball_y_mm)
                self.ball_detected = ball_data['detected']

                if self.ball_detected:
                    self.ball_history_x.append(ball_x_mm)
                    self.ball_history_y.append(ball_y_mm)
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

        if self.ball_detected:
            status = "Detected"
        else:
            status = "Not detected"

        self.ball_pos_label.config(
            text=f"Position: {format_vector_2d(self.ball_pos_mm)}"
        )
        self.ball_vel_label.config(text=f"Status: {status}")

        if self.controller_enabled.get():
            rx = self.dof_values['rx']
            ry = self.dof_values['ry']

            self.value_labels['rx'].config(text=f"{rx:.2f}")
            self.value_labels['ry'].config(text=f"{ry:.2f}")

            pattern_time = self.simulation_time - self.pattern_start_time
            target_x, target_y = self.current_pattern.get_position(pattern_time)
            error_x = target_x - self.ball_pos_mm[0]
            error_y = target_y - self.ball_pos_mm[1]

            magnitude = np.sqrt(rx ** 2 + ry ** 2)
            magnitude_percent = (magnitude / 15.0) * 100

            self.controller_output_label.config(text=f"Tilt: rx={rx:.2f}°  ry={ry:.2f}°")
            self.tilt_magnitude_label.config(
                text=f"Magnitude: {magnitude:.2f}° ({magnitude_percent:.1f}%)"
            )
            self.controller_error_label.config(
                text=f"Error: {format_vector_2d((error_x, error_y))}"
            )

        if self.last_sent_angles is not None:
            for i in range(6):
                self.cmd_angle_labels[i].config(
                    text=f"S{i + 1}: {self.last_sent_angles[i]:6.2f}°"
                )

        self.sim_time_label.config(text=f"Time: {format_time(self.simulation_time)}")

        self.gui_update_count += 1
        if self.gui_update_count % 2 == 0:
            self._update_hardware_plot()

        current_time = time.time()
        if current_time - self.last_gui_update >= 1.0:
            self.fps_label.config(text=f"Control: {ControlLoopConfig.FREQUENCY_HZ:.1f} Hz")

            hit_rate = self.ik_cache.get_hit_rate()
            self.cache_label.config(text=f"IK Cache: {hit_rate * 100:.1f}%")

            self.timeout_label.config(text=f"IK Timeouts: {self.ik_timeout_count}")

            self.last_gui_update = current_time

        self.root.after(GUIConfig.UPDATE_INTERVAL_MS, self._gui_update_loop)

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
        stats_msg = "Performance Statistics (100Hz Hardware Mode)\n"
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

        stats_msg += "Optimization:\n"
        stats_msg += f"  IK Timeouts: {self.ik_timeout_count}\n"
        stats_msg += f"  Cache Resolution: 1mm / 1°\n"
        stats_msg += f"  Target Rate: {ControlLoopConfig.FREQUENCY_HZ} Hz\n"
        stats_msg += f"  GUI Update: {GUIConfig.UPDATE_HZ} Hz\n"

        messagebox.showinfo("Performance Statistics", stats_msg)

    def _update_controller(self, ball_pos_mm, ball_vel_mm_s, target_pos_mm, dt):
        """Hardware controller update (compatibility with base class)."""
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

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log("⏸ Control stopped")

    def on_closing(self):
        """Clean shutdown."""
        if self.simulation_running:
            self.stop_simulation()

        if self.connected:
            self.disconnect_serial()

        super().on_closing()


def main():
    """Launch hardware controller."""
    root = tk.Tk()
    app = HardwareStewartSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()