#!/usr/bin/env python3
"""
Stewart Platform Real Hardware Controller with PID Ball Balancing
NON-BLOCKING VERSION - Fixes freezing issues
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np
import time
import threading
import serial
import serial.tools.list_ports
from queue import Queue, Empty
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle

from core import StewartPlatformIK, PatternFactory
from control_core import PIDController


class SerialController:
    """Handles serial communication with Arduino - NON-BLOCKING."""

    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        self.read_thread = None
        self.running = False

        # Queues for thread-safe communication
        self.ball_data_queue = Queue(maxsize=10)
        self.command_queue = Queue(maxsize=50)  # Buffer commands

        # Write thread for non-blocking writes
        self.write_thread = None

        # Rate limiting
        self.last_command_time = 0
        self.min_command_interval = 0.01  # 100Hz max command rate

    def connect(self):
        """Connect to Arduino."""
        try:
            # Reduced timeout for non-blocking operation
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1, write_timeout=0.1)
            time.sleep(2)  # Arduino reset delay
            self.connected = True

            # Clear any startup messages
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

            # Start threads
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
        """Disconnect from Arduino."""
        self.running = False

        if self.read_thread:
            self.read_thread.join(timeout=1)
        if self.write_thread:
            self.write_thread.join(timeout=1)

        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False

    def _read_loop(self):
        """Background thread for reading serial data - NON-BLOCKING."""
        buffer = ""

        while self.running and self.serial and self.serial.is_open:
            try:
                # Non-blocking read
                if self.serial.in_waiting > 0:
                    chunk = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk

                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if line.startswith("BALL:"):
                            # Parse: timestamp,x,y,detected,error_x,error_y
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

                                    # Add to queue (drop old if full)
                                    if self.ball_data_queue.full():
                                        try:
                                            self.ball_data_queue.get_nowait()
                                        except Empty:
                                            pass
                                    self.ball_data_queue.put(ball_data)
                            except (ValueError, IndexError):
                                pass  # Ignore malformed data

                # Don't hog CPU
                time.sleep(0.001)

            except Exception as e:
                if self.running:  # Only log if not shutting down
                    print(f"Serial read error: {e}")
                time.sleep(0.1)

    def _write_loop(self):
        """Background thread for writing serial data - NON-BLOCKING."""
        while self.running and self.serial and self.serial.is_open:
            try:
                # Get command from queue (non-blocking)
                try:
                    command = self.command_queue.get(timeout=0.05)

                    # Rate limiting
                    now = time.time()
                    elapsed = now - self.last_command_time
                    if elapsed < self.min_command_interval:
                        time.sleep(self.min_command_interval - elapsed)

                    # Send command
                    self.serial.write(command.encode('utf-8'))
                    self.serial.flush()
                    self.last_command_time = time.time()

                except Empty:
                    pass  # No command to send

            except Exception as e:
                if self.running:
                    print(f"Serial write error: {e}")
                time.sleep(0.1)

    def send_servo_angles(self, angles):
        """Queue servo angles to send (non-blocking)."""
        if not self.connected:
            return False

        try:
            command = ','.join([f'{angle:.2f}' for angle in angles]) + '\n'

            # If queue is full, drop oldest command
            if self.command_queue.full():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    pass

            self.command_queue.put(command)
            return True
        except Exception as e:
            print(f"Error queueing command: {e}")
            return False

    def send_command(self, cmd):
        """Send a text command."""
        if not self.connected:
            return False

        try:
            if self.command_queue.full():
                try:
                    self.command_queue.get_nowait()
                except Empty:
                    pass
            self.command_queue.put(cmd + '\n')
            return True
        except:
            return False

    def set_servo_speed(self, speed):
        """Set servo speed (0-255)."""
        self.send_command(f'SPD:{speed}')

    def set_servo_acceleration(self, accel):
        """Set servo acceleration (0-255)."""
        self.send_command(f'ACC:{accel}')

    def get_latest_ball_data(self):
        """Get most recent ball position data."""
        try:
            data = None
            while not self.ball_data_queue.empty():
                data = self.ball_data_queue.get_nowait()
            return data
        except Empty:
            return None


class StewartRealControllerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stewart Platform - Real Hardware PID Control")
        self.root.geometry("1400x900")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Dark mode colors
        self.colors = {
            'bg': '#1e1e1e',
            'panel_bg': '#2d2d2d',
            'widget_bg': '#3d3d3d',
            'fg': '#e0e0e0',
            'highlight': '#007acc',
            'button_bg': '#0e639c',
            'button_fg': '#ffffff',
            'entry_bg': '#3d3d3d',
            'border': '#555555',
            'success': '#4ec9b0',
            'warning': '#ce9178'
        }

        self.root.configure(bg=self.colors['bg'])
        self.setup_dark_theme()

        # Platform parameters
        self.platform_params = {
            "horn_length": 31.75,
            "rod_length": 145.0,
            "base": 73.025,
            "base_anchors": 36.8893,
            "platform": 67.775,
            "platform_anchors": 12.7,
            "top_surface_offset": 26.0
        }
        self.ik = StewartPlatformIK(**self.platform_params)

        # Camera calibration
        self.pixy_width_mm = 200.0
        self.pixy_height_mm = 133.0
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0

        # Serial communication
        self.serial_controller = None
        self.connected = False

        # Ball state
        self.ball_pos_mm = (0.0, 0.0)
        self.ball_detected = False
        self.last_ball_update = 0

        # PID Controller
        self.pid_controller = PIDController(kp=0.003, ki=0.001, kd=0.003, output_limit=15.0)
        self.pid_enabled = tk.BooleanVar(value=False)

        # Trajectory pattern
        self.current_pattern = PatternFactory.create('static', x=0.0, y=0.0)
        self.pattern_type = tk.StringVar(value='static')
        self.pattern_start_time = 0.0

        # PID gain scalars
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        self.kp_scalar_idx = tk.IntVar(value=4)
        self.ki_scalar_idx = tk.IntVar(value=4)
        self.kd_scalar_idx = tk.IntVar(value=4)

        # Control state
        self.control_running = False
        self.control_time = 0.0
        self.last_control_update = None
        self.control_loop_id = None
        self.update_rate_ms = 50  # 20Hz control loop

        # Platform pose
        self.use_top_surface_offset = tk.BooleanVar(value=True)
        self.dof_values = {
            'x': 0.0, 'y': 0.0, 'z': self.ik.home_height_top_surface,
            'rx': 0.0, 'ry': 0.0, 'rz': 0.0
        }

        self.dof_config = {
            'x': (-30.0, 30.0, 0.1, 0.0, "X Position (mm)"),
            'y': (-30.0, 30.0, 0.1, 0.0, "Y Position (mm)"),
            'z': (self.ik.home_height_top_surface - 30,
                  self.ik.home_height_top_surface + 30,
                  0.1, self.ik.home_height_top_surface, "Z Height (mm)"),
            'rx': (-15.0, 15.0, 0.1, 0.0, "Roll (°)"),
            'ry': (-15.0, 15.0, 0.1, 0.0, "Pitch (°)"),
            'rz': (-15.0, 15.0, 0.1, 0.0, "Yaw (°)")
        }

        self.sliders = {}
        self.value_labels = {}
        self.pid_sliders = {}
        self.pid_value_labels = {}
        self.update_timer = None

        # Ball position history
        self.ball_history_x = []
        self.ball_history_y = []
        self.max_history = 100

        # Last sent angles to prevent duplicate sends
        self.last_sent_angles = None

        self.create_widgets()

    def setup_dark_theme(self):
        """Configure ttk widgets for dark mode."""
        style = ttk.Style()
        style.theme_use('default')

        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Card.TFrame', background=self.colors['panel_bg'], relief='flat')

        style.configure('TLabelframe',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        borderwidth=1,
                        relief='solid')
        style.configure('TLabelframe.Label',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['highlight'],
                        font=('Segoe UI', 9, 'bold'))

        style.configure('TLabel',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        font=('Segoe UI', 9))

        style.configure('TButton',
                        background=self.colors['button_bg'],
                        foreground=self.colors['button_fg'],
                        borderwidth=0,
                        focuscolor='none',
                        font=('Segoe UI', 9))
        style.map('TButton',
                  background=[('active', self.colors['highlight']),
                              ('pressed', '#005a9e')])

        style.configure('TScale',
                        background=self.colors['panel_bg'],
                        troughcolor=self.colors['widget_bg'],
                        borderwidth=0)

        style.configure('TCheckbutton',
                        background=self.colors['panel_bg'],
                        foreground=self.colors['fg'],
                        font=('Segoe UI', 9))

        style.configure('TCombobox',
                        fieldbackground=self.colors['widget_bg'],
                        background=self.colors['button_bg'],
                        foreground=self.colors['fg'],
                        arrowcolor=self.colors['fg'])
        style.map('TCombobox',
                  fieldbackground=[('readonly', self.colors['widget_bg'])],
                  selectbackground=[('readonly', self.colors['highlight'])],
                  selectforeground=[('readonly', self.colors['button_fg'])])

        # Configure combobox dropdown list (Listbox widget)
        self.root.option_add('*TCombobox*Listbox.background', self.colors['widget_bg'])
        self.root.option_add('*TCombobox*Listbox.foreground', self.colors['fg'])
        self.root.option_add('*TCombobox*Listbox.selectBackground', self.colors['highlight'])
        self.root.option_add('*TCombobox*Listbox.selectForeground', self.colors['button_fg'])
        self.root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 9))

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        left_panel = ttk.Frame(main_frame, style='TFrame', width=400)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 5))
        left_panel.pack_propagate(False)

        middle_panel = ttk.Frame(main_frame, style='TFrame', width=450)
        middle_panel.pack(side='left', fill='both', expand=False, padx=5)
        middle_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_frame, style='TFrame')
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))

        # === LEFT PANEL ===

        # Serial connection
        conn_frame = ttk.LabelFrame(left_panel, text="Serial Connection", padding=10)
        conn_frame.pack(fill='x', pady=(0, 10))

        port_frame = ttk.Frame(conn_frame)
        port_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(port_frame, text="Port:").pack(side='left', padx=(0, 5))

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=15, state='readonly')
        self.port_combo.pack(side='left', padx=5)
        self.refresh_ports()

        ttk.Button(port_frame, text="↻", command=self.refresh_ports, width=3).pack(side='left')

        conn_btn_frame = ttk.Frame(conn_frame)
        conn_btn_frame.pack(fill='x', pady=(5, 0))

        self.connect_btn = ttk.Button(conn_btn_frame, text="Connect", command=self.connect_serial, width=12)
        self.connect_btn.pack(side='left', padx=5)

        self.disconnect_btn = ttk.Button(conn_btn_frame, text="Disconnect", command=self.disconnect_serial,
                                         state='disabled', width=12)
        self.disconnect_btn.pack(side='left', padx=5)

        self.conn_status_label = ttk.Label(conn_frame, text="Not connected", foreground=self.colors['border'])
        self.conn_status_label.pack(pady=(5, 0))

        # Control
        ctrl_frame = ttk.LabelFrame(left_panel, text="Control", padding=10)
        ctrl_frame.pack(fill='x', pady=(0, 10))

        ctrl_btn_frame = ttk.Frame(ctrl_frame)
        ctrl_btn_frame.pack(fill='x')

        self.start_btn = ttk.Button(ctrl_btn_frame, text="▶ Start", command=self.start_control,
                                    state='disabled', width=10)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(ctrl_btn_frame, text="⏸ Stop", command=self.stop_control,
                                   state='disabled', width=10)
        self.stop_btn.pack(side='left', padx=5)

        self.home_btn = ttk.Button(ctrl_btn_frame, text="⌂ Home", command=self.go_home,
                                   state='disabled', width=10)
        self.home_btn.pack(side='left', padx=5)

        self.control_time_label = ttk.Label(ctrl_frame, text="Time: 0.00s", font=('Consolas', 10, 'bold'))
        self.control_time_label.pack(pady=(10, 0))

        # Camera calibration
        cal_frame = ttk.LabelFrame(left_panel, text="Camera Calibration", padding=10)
        cal_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(cal_frame, text="Visible width (mm):", font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w', pady=2)
        self.width_entry = ttk.Entry(cal_frame, width=10)
        self.width_entry.insert(0, str(self.pixy_width_mm))
        self.width_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(cal_frame, text="Visible height (mm):", font=('Segoe UI', 9)).grid(row=1, column=0, sticky='w',
                                                                                     pady=2)
        self.height_entry = ttk.Entry(cal_frame, width=10)
        self.height_entry.insert(0, str(self.pixy_height_mm))
        self.height_entry.grid(row=1, column=1, padx=5, pady=2)

        ttk.Button(cal_frame, text="Update", command=self.update_calibration, width=10).grid(row=2, column=0,
                                                                                             columnspan=2, pady=5)

        # PID Control
        pid_frame = ttk.LabelFrame(left_panel, text="PID Ball Balancing", padding=10)
        pid_frame.pack(fill='x', pady=(0, 10))

        enable_frame = ttk.Frame(pid_frame)
        enable_frame.pack(fill='x', pady=(0, 10))

        ttk.Checkbutton(enable_frame, text="Enable PID Control",
                        variable=self.pid_enabled,
                        command=self.on_pid_toggle).pack(side='left')

        self.pid_status_label = ttk.Label(enable_frame, text="●",
                                          foreground=self.colors['border'],
                                          font=('Segoe UI', 14))
        self.pid_status_label.pack(side='left', padx=(10, 0))

        # PID gains
        pid_gains = [('kp', 'P (Proportional)', 3.0),
                     ('ki', 'I (Integral)', 1.0),
                     ('kd', 'D (Derivative)', 3.0)]

        pid_defaults = {}

        for gain_name, label, default in pid_gains:
            pid_defaults[gain_name] = default

            frame = ttk.Frame(pid_frame)
            frame.pack(fill='x', pady=5)

            ttk.Label(frame, text=label, font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w', pady=2)

            slider = ttk.Scale(frame, from_=0.0, to=10.0, orient='horizontal')
            slider.grid(row=0, column=1, sticky='ew', padx=10)
            self.pid_sliders[gain_name] = slider

            value_label = ttk.Label(frame, text=f"{default:.2f}", width=6, font=('Consolas', 9))
            value_label.grid(row=0, column=2)
            self.pid_value_labels[gain_name] = value_label

            scalar_combo = ttk.Combobox(frame, width=12, state='readonly',
                                        values=[f'×{s:.7g}' for s in self.scalar_values])
            scalar_combo.grid(row=0, column=3, padx=(5, 0))

            scalar_var = getattr(self, f'{gain_name}_scalar_idx')
            scalar_combo.current(scalar_var.get())

            scalar_combo.bind('<<ComboboxSelected>>',
                              lambda e, combo=scalar_combo, var=scalar_var, g=gain_name:
                              self._on_scalar_selected(combo, var, g))

            frame.columnconfigure(1, weight=1)

        for gain_name, default in pid_defaults.items():
            self.pid_sliders[gain_name].set(default)
            self.pid_sliders[gain_name].config(
                command=lambda val, g=gain_name: self.on_pid_slider_change(g, val)
            )

        self.update_pid_gains()

        # Trajectory pattern
        pattern_frame = ttk.LabelFrame(pid_frame, text="Trajectory Pattern", padding=10)
        pattern_frame.pack(fill='x', pady=(10, 0))

        selector_frame = ttk.Frame(pattern_frame)
        selector_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(selector_frame, text="Pattern:", font=('Segoe UI', 9)).pack(side='left', padx=(0, 10))

        pattern_combo = ttk.Combobox(selector_frame, textvariable=self.pattern_type,
                                     width=15, state='readonly',
                                     values=['static', 'circle', 'figure8', 'star'])
        pattern_combo.pack(side='left', padx=5)
        pattern_combo.bind('<<ComboboxSelected>>', self.on_pattern_change)

        ttk.Button(selector_frame, text="Reset", command=self.reset_pattern, width=8).pack(side='left', padx=5)

        self.pattern_info_label = ttk.Label(pattern_frame,
                                            text="Tracking: Center (0, 0)",
                                            font=('Consolas', 8),
                                            foreground=self.colors['success'])
        self.pattern_info_label.pack(anchor='w', pady=(5, 0))

        # Ball state
        ball_info_frame = ttk.LabelFrame(left_panel, text="Ball State", padding=10)
        ball_info_frame.pack(fill='x', pady=(0, 10))

        self.ball_pos_label = ttk.Label(ball_info_frame, text="Position: (0.0, 0.0) mm", font=('Consolas', 9))
        self.ball_pos_label.pack(anchor='w', pady=2)

        self.ball_status_label = ttk.Label(ball_info_frame, text="Status: Not detected", font=('Consolas', 9))
        self.ball_status_label.pack(anchor='w', pady=2)

        # Manual sliders
        sliders_frame = ttk.LabelFrame(left_panel, text="Manual Pose Control (6 DOF)", padding=10)
        sliders_frame.pack(fill='both', expand=True)

        for idx, (dof, (min_val, max_val, res, default, label)) in enumerate(self.dof_config.items()):
            ttk.Label(sliders_frame, text=label, font=('Segoe UI', 9)).grid(row=idx, column=0, sticky='w', pady=8)

            slider = ttk.Scale(sliders_frame, from_=min_val, to=max_val, orient='horizontal',
                               command=lambda val, d=dof: self.on_slider_change(d, val))
            slider.grid(row=idx, column=1, sticky='ew', padx=10, pady=8)
            self.sliders[dof] = slider

            value_label = ttk.Label(sliders_frame, text=f"{default:.2f}", width=8, font=('Consolas', 9))
            value_label.grid(row=idx, column=2, pady=8)
            self.value_labels[dof] = value_label
            slider.set(default)

        sliders_frame.columnconfigure(1, weight=1)

        # === MIDDLE PANEL ===

        # Commanded angles
        cmd_angles_frame = ttk.LabelFrame(middle_panel, text="Commanded Servo Angles", padding=10)
        cmd_angles_frame.pack(fill='x', pady=(0, 10))

        self.cmd_angle_labels = []
        for i in range(6):
            label = ttk.Label(cmd_angles_frame, text=f"S{i + 1}: 0.00°", font=('Consolas', 9))
            label.grid(row=i // 3, column=i % 3, padx=15, pady=5, sticky='w')
            self.cmd_angle_labels.append(label)

        # PID output
        pid_output_frame = ttk.LabelFrame(middle_panel, text="PID Output", padding=10)
        pid_output_frame.pack(fill='x', pady=(0, 10))

        self.pid_output_label = ttk.Label(pid_output_frame, text="Tilt: rx=0.00°  ry=0.00°", font=('Consolas', 9))
        self.pid_output_label.pack(anchor='w', pady=2)

        self.pid_error_label = ttk.Label(pid_output_frame, text="Error: (0.0, 0.0) mm", font=('Consolas', 9))
        self.pid_error_label.pack(anchor='w', pady=2)

        # Log
        log_frame = ttk.LabelFrame(middle_panel, text="System Log", padding=10)
        log_frame.pack(fill='both', expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=20,
            font=('Consolas', 8),
            bg=self.colors['widget_bg'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg'],
            selectbackground=self.colors['highlight'],
            selectforeground=self.colors['button_fg'],
            relief='flat',
            borderwidth=0
        )
        self.log_text.pack(fill='both', expand=True)

        # === RIGHT PANEL ===
        plot_frame = ttk.LabelFrame(right_panel, text="Ball Position (Top View)", padding=10)
        plot_frame.pack(fill='both', expand=True)

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 6), facecolor=self.colors['panel_bg'])
        self.ax.set_facecolor(self.colors['widget_bg'])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.setup_plot()
        self.log("Real hardware controller initialized (non-blocking)")
        self.log("Connect to Teensy to begin")

    def setup_plot(self):
        """Setup the matplotlib plot."""
        self.ax.clear()
        self.ax.set_xlim(-120, 120)
        self.ax.set_ylim(-120, 120)
        self.ax.set_xlabel('X (mm)', color=self.colors['fg'], fontsize=10)
        self.ax.set_ylabel('Y (mm)', color=self.colors['fg'], fontsize=10)
        self.ax.set_title('Ball Position (Top View)', color=self.colors['fg'], fontsize=11, fontweight='bold')
        self.ax.grid(True, alpha=0.2, linestyle='--', color=self.colors['fg'])
        self.ax.set_aspect('equal')

        self.ax.tick_params(colors=self.colors['fg'])

        for spine in self.ax.spines.values():
            spine.set_color(self.colors['border'])

        platform_square = Rectangle((-100, -100), 200, 200,
                                    fill=False,
                                    edgecolor=self.colors['fg'],
                                    linewidth=2,
                                    linestyle='--',
                                    label='Platform Edge',
                                    alpha=0.5)
        self.ax.add_patch(platform_square)

        self.trajectory_line, = self.ax.plot([], [], '--', color=self.colors['highlight'],
                                             alpha=0.3, linewidth=1, label='Trajectory')
        self.target_marker, = self.ax.plot([0], [0], 'x', color=self.colors['success'],
                                           markersize=10, markeredgewidth=2, label='Target')

        self.ball_circle = Circle((0, 0), 3.0, color='#ff4444', alpha=0.8,
                                  zorder=10, label='Ball')
        self.ax.add_patch(self.ball_circle)

        self.ball_trail, = self.ax.plot([], [], '-', color='#ff4444', alpha=0.3, linewidth=1, label='Ball Trail')

        self.tilt_arrow = None

        legend = self.ax.legend(loc='upper right', fontsize=8, facecolor=self.colors['panel_bg'],
                                edgecolor=self.colors['border'], labelcolor=self.colors['fg'])
        legend.get_frame().set_alpha(0.9)

        self.canvas.draw()

    def refresh_ports(self):
        """Refresh list of available serial ports."""
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def connect_serial(self):
        """Connect to Teensy."""
        port = self.port_var.get()
        if not port:
            messagebox.showerror("Error", "Please select a serial port")
            return

        self.serial_controller = SerialController(port)
        success, message = self.serial_controller.connect()

        if success:
            self.connected = True
            self.conn_status_label.config(text="Connected", foreground=self.colors['success'])
            self.connect_btn.config(state='disabled')
            self.disconnect_btn.config(state='normal')
            self.start_btn.config(state='normal')
            self.home_btn.config(state='normal')
            self.log(f"Connected to {port}")
        else:
            messagebox.showerror("Connection Error", message)
            self.log(f"Failed to connect: {message}")

    def disconnect_serial(self):
        """Disconnect from Teensy."""
        if self.control_running:
            self.stop_control()

        if self.serial_controller:
            self.serial_controller.disconnect()

        self.connected = False
        self.conn_status_label.config(text="Not connected", foreground=self.colors['border'])
        self.connect_btn.config(state='normal')
        self.disconnect_btn.config(state='disabled')
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='disabled')
        self.home_btn.config(state='disabled')
        self.log("Disconnected")

    def update_calibration(self):
        """Update camera calibration."""
        try:
            width = float(self.width_entry.get())
            height = float(self.height_entry.get())

            self.pixy_width_mm = width
            self.pixy_height_mm = height
            self.pixels_to_mm_x = self.pixy_width_mm / 316.0
            self.pixels_to_mm_y = self.pixy_height_mm / 208.0

            self.log(f"Calibration updated: {width}×{height}mm")
            self.log(f"Scale: {self.pixels_to_mm_x:.3f} mm/px (X), {self.pixels_to_mm_y:.3f} mm/px (Y)")
        except ValueError:
            messagebox.showerror("Error", "Invalid calibration values")

    def go_home(self):
        """Move platform to home position."""
        if not self.connected:
            return

        self.dof_values['x'] = 0.0
        self.dof_values['y'] = 0.0
        self.dof_values['z'] = self.ik.home_height_top_surface
        self.dof_values['rx'] = 0.0
        self.dof_values['ry'] = 0.0
        self.dof_values['rz'] = 0.0

        for dof in ['x', 'y', 'rx', 'ry', 'rz']:
            self.sliders[dof].set(0.0)
            self.value_labels[dof].config(text="0.00")

        self.sliders['z'].set(self.ik.home_height_top_surface)
        self.value_labels['z'].config(text=f"{self.ik.home_height_top_surface:.2f}")

        self.calculate_and_send_ik()
        self.log("Moved to home position")

    def start_control(self):
        """Start control loop."""
        if not self.connected:
            return

        self.control_running = True
        self.last_control_update = time.time()
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.log("Control started")
        self.control_loop()

    def stop_control(self):
        """Stop control loop."""
        self.control_running = False

        if self.control_loop_id is not None:
            self.root.after_cancel(self.control_loop_id)
            self.control_loop_id = None

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log("Control stopped")

    def control_loop(self):
        """Main control loop."""
        if not self.control_running or not self.connected:
            self.control_loop_id = None
            return

        current_time = time.time()
        if self.last_control_update is not None:
            dt = current_time - self.last_control_update
            self.control_time += dt

            # Get latest ball position
            ball_data = self.serial_controller.get_latest_ball_data()

            if ball_data is not None:
                self.last_ball_update = self.control_time

                # Convert Pixy coordinates to mm
                pixy_x = ball_data['x']
                pixy_y = ball_data['y']

                ball_x_mm = (pixy_x - 158.0) * self.pixels_to_mm_x
                ball_y_mm = -(pixy_y - 104.0) * self.pixels_to_mm_y

                self.ball_pos_mm = (ball_x_mm, ball_y_mm)
                self.ball_detected = ball_data['detected']

                # Update ball history
                if self.ball_detected:
                    self.ball_history_x.append(ball_x_mm)
                    self.ball_history_y.append(ball_y_mm)
                    if len(self.ball_history_x) > self.max_history:
                        self.ball_history_x.pop(0)
                        self.ball_history_y.pop(0)

                # Update display
                status = "Detected" if self.ball_detected else "Not detected"
                self.ball_pos_label.config(text=f"Position: ({ball_x_mm:.1f}, {ball_y_mm:.1f}) mm")
                self.ball_status_label.config(text=f"Status: {status}")

            # PID control
            if self.pid_enabled.get() and self.ball_detected:
                pattern_time = self.control_time - self.pattern_start_time
                target_x, target_y = self.current_pattern.get_position(pattern_time)
                target_pos_mm = (target_x, target_y)

                rx, ry = self.pid_controller.update(self.ball_pos_mm, target_pos_mm, dt)
                ry = -ry

                self.dof_values['rx'] = rx
                self.dof_values['ry'] = ry

                self.value_labels['rx'].config(text=f"{rx:.2f}")
                self.value_labels['ry'].config(text=f"{ry:.2f}")

                # Update PID output display
                error_x = target_pos_mm[0] - self.ball_pos_mm[0]
                error_y = target_pos_mm[1] - self.ball_pos_mm[1]
                self.pid_output_label.config(text=f"Tilt: rx={rx:.2f}°  ry={ry:.2f}°")
                self.pid_error_label.config(text=f"Error: ({error_x:.1f}, {error_y:.1f}) mm")

                # Send to platform
                self.calculate_and_send_ik()

            # Update plot
            self.update_plot()

        self.last_control_update = current_time
        self.control_time_label.config(text=f"Time: {self.control_time:.2f}s")

        # Schedule next iteration
        self.control_loop_id = self.root.after(self.update_rate_ms, self.control_loop)

    def calculate_and_send_ik(self):
        """Calculate IK and send servo angles."""
        translation = np.array([self.dof_values['x'], self.dof_values['y'], self.dof_values['z']])
        rotation = np.array([self.dof_values['rx'], self.dof_values['ry'], self.dof_values['rz']])

        angles = self.ik.calculate_servo_angles(translation, rotation, self.use_top_surface_offset.get())

        if angles is not None:
            # Only send if changed (prevent flooding)
            if self.last_sent_angles is None or not np.allclose(angles, self.last_sent_angles, atol=0.1):
                # Update display
                for i in range(6):
                    self.cmd_angle_labels[i].config(text=f"S{i + 1}: {angles[i]:6.2f}°")

                # Send to Teensy
                if self.serial_controller:
                    self.serial_controller.send_servo_angles(angles)
                    self.last_sent_angles = angles.copy()
        else:
            self.log("IK solution out of range")

    def update_plot(self):
        """Update the plot."""
        if self.ball_detected:
            self.ball_circle.center = self.ball_pos_mm
            self.ball_circle.set_alpha(0.8)
        else:
            self.ball_circle.set_alpha(0.2)

        if len(self.ball_history_x) > 1:
            self.ball_trail.set_data(self.ball_history_x, self.ball_history_y)

        if self.pattern_type.get() != 'static':
            pattern_time = self.control_time - self.pattern_start_time

            pattern_periods = {
                'circle': 10.0,
                'figure8': 12.0,
                'star': 15.0
            }
            period = pattern_periods.get(self.pattern_type.get(), 10.0)

            t_samples = np.linspace(0, period, 100)
            path_x = []
            path_y = []
            for t in t_samples:
                x, y = self.current_pattern.get_position(t)
                path_x.append(x)
                path_y.append(y)

            self.trajectory_line.set_data(path_x, path_y)

            target_x, target_y = self.current_pattern.get_position(pattern_time)
            self.target_marker.set_data([target_x], [target_y])
        else:
            self.trajectory_line.set_data([], [])
            self.target_marker.set_data([0], [0])

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
                color = self.colors['success'] if self.pid_enabled.get() else self.colors['highlight']
                self.tilt_arrow = self.ax.arrow(0, 0, dx, dy, head_width=8, head_length=10,
                                                fc=color, ec=color,
                                                alpha=0.6, linewidth=2, zorder=5)

        self.canvas.draw_idle()

    def on_pid_toggle(self):
        """Handle PID enable/disable toggle."""
        enabled = self.pid_enabled.get()

        if enabled:
            if not self.connected:
                messagebox.showwarning("Warning", "Not connected to Teensy")
                self.pid_enabled.set(False)
                return

            self.pid_controller.reset()
            self.reset_pattern()
            self.pid_status_label.config(foreground=self.colors['success'])

            kp = self.pid_controller.kp
            ki = self.pid_controller.ki
            kd = self.pid_controller.kd
            self.log("PID control ENABLED")
            self.log(f"Gains: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

            self.sliders['rx'].config(state='disabled')
            self.sliders['ry'].config(state='disabled')

            self.dof_values['rx'] = 0.0
            self.dof_values['ry'] = 0.0
            self.value_labels['rx'].config(text="0.00")
            self.value_labels['ry'].config(text="0.00")
        else:
            self.pid_status_label.config(foreground=self.colors['border'])
            self.log("PID control DISABLED")

            self.sliders['rx'].config(state='normal')
            self.sliders['ry'].config(state='normal')

    def on_pid_slider_change(self, gain_name, value):
        """Handle PID gain slider changes."""
        val = float(value)
        self.pid_value_labels[gain_name].config(text=f"{val:.2f}")
        self.update_pid_gains()

    def _on_scalar_selected(self, combo, int_var, gain_name):
        """Handle scalar combobox selection."""
        selected_index = combo.current()
        int_var.set(selected_index)
        self.update_pid_gains()

    def update_pid_gains(self):
        """Update PID controller with current gain values."""
        kp_raw = float(self.pid_sliders['kp'].get())
        ki_raw = float(self.pid_sliders['ki'].get())
        kd_raw = float(self.pid_sliders['kd'].get())

        kp_idx = max(0, min(self.kp_scalar_idx.get(), len(self.scalar_values) - 1))
        ki_idx = max(0, min(self.ki_scalar_idx.get(), len(self.scalar_values) - 1))
        kd_idx = max(0, min(self.kd_scalar_idx.get(), len(self.scalar_values) - 1))

        kp = kp_raw * self.scalar_values[kp_idx]
        ki = ki_raw * self.scalar_values[ki_idx]
        kd = kd_raw * self.scalar_values[kd_idx]

        self.pid_controller.set_gains(kp, ki, kd)

        if self.pid_enabled.get():
            self.log(f"PID gains updated: Kp={kp:.6f}, Ki={ki:.6f}, Kd={kd:.6f}")

    def on_pattern_change(self, event=None):
        """Handle pattern selection change."""
        pattern_type = self.pattern_type.get()

        if pattern_type == 'static':
            self.current_pattern = PatternFactory.create('static', x=0.0, y=0.0)
            info = "Tracking: Center (0, 0)"
        elif pattern_type == 'circle':
            self.current_pattern = PatternFactory.create('circle', radius=50.0, period=10.0, clockwise=True)
            info = "Tracking: Circle (r=50mm, T=10s)"
        elif pattern_type == 'figure8':
            self.current_pattern = PatternFactory.create('figure8', width=60.0, height=40.0, period=12.0)
            info = "Tracking: Figure-8 (60×40mm, T=12s)"
        elif pattern_type == 'star':
            self.current_pattern = PatternFactory.create('star', radius=60.0, period=15.0)
            info = "Tracking: 5-Point Star (r=60mm, T=15s)"
        else:
            return

        self.pattern_info_label.config(text=info)
        self.reset_pattern()
        self.update_plot()
        self.log(f"Pattern changed to: {pattern_type}")

    def reset_pattern(self):
        """Reset pattern."""
        self.pattern_start_time = self.control_time
        self.current_pattern.reset()
        self.log(f"Pattern reset at t={self.control_time:.2f}s")

        if self.pid_enabled.get():
            self.pid_controller.reset()

    def on_slider_change(self, dof, value):
        """Handle manual slider changes."""
        val = float(value)
        self.dof_values[dof] = val
        self.value_labels[dof].config(text=f"{val:.2f}")

        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(50, self.calculate_and_send_ik)

    def log(self, message):
        """Add message to log."""
        timestamp = f"[{self.control_time:.2f}s]" if self.control_running else "[--]"
        self.log_text.insert(tk.END, f"{timestamp} {message}\n")
        self.log_text.see(tk.END)

    def on_closing(self):
        """Clean shutdown."""
        if self.control_running:
            self.stop_control()

        if self.connected:
            self.disconnect_serial()

        if self.control_loop_id is not None:
            try:
                self.root.after_cancel(self.control_loop_id)
            except:
                pass

        if self.update_timer is not None:
            try:
                self.root.after_cancel(self.update_timer)
            except:
                pass

        try:
            self.root.quit()
            self.root.destroy()
        except:
            pass


def main():
    root = tk.Tk()
    app = StewartRealControllerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()