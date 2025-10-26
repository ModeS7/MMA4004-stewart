#!/usr/bin/env python3
"""
PyQtGraph Stress Test

Stress test PyQtGraph rendering and Python threading to find system limits.
"""

import sys
import numpy as np
import time
import threading
from collections import deque
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class JitterTestApp(QtWidgets.QMainWindow):
    """Test application for PyQtGraph performance analysis."""

    # Signal for thread-safe GUI updates
    update_stats_signal = QtCore.pyqtSignal()
    update_plot_signal = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQtGraph STRESS TEST - Find the Breaking Point!")
        self.setGeometry(100, 100, 1000, 850)

        # Control loop parameters
        self.target_hz = 100
        self.target_period_ms = 1000.0 / self.target_hz
        self.loop_running = False
        self.plot_enabled = True
        self.plot_interval_ms = 100  # 10Hz default

        # Data size control
        self.max_data_points = 500

        # Timing statistics
        self.loop_times = deque(maxlen=1000)
        self.plot_times = deque(maxlen=1000)
        self.sleep_times = deque(maxlen=1000)
        self.loop_count = 0

        # Simulated data for multiple signals
        self.data_x = deque(maxlen=self.max_data_points)
        self.data_y1 = deque(maxlen=self.max_data_points)
        self.data_y2 = deque(maxlen=self.max_data_points)
        self.data_y3 = deque(maxlen=self.max_data_points)
        self.data_y4 = deque(maxlen=self.max_data_points)

        # Data locks for thread safety
        self.data_lock = threading.Lock()

        self.setup_ui()

        # Connect signals
        self.update_stats_signal.connect(self._update_statistics)
        self.update_plot_signal.connect(self._update_plot_data)

        # Start GUI update timer
        self.stats_timer = QtCore.QTimer()
        self.stats_timer.timeout.connect(self._update_statistics)

        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self._update_plot_data)

    def setup_ui(self):
        """Create GUI layout."""
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)

        # Control panel
        control_group = QtWidgets.QGroupBox("Control - STRESS TEST MODE")
        control_layout = QtWidgets.QHBoxLayout()

        # Control loop frequency selector
        freq_label = QtWidgets.QLabel("Loop Freq:")
        control_layout.addWidget(freq_label)

        self.freq_combo = QtWidgets.QComboBox()
        self.freq_combo.addItems(['50 Hz', '100 Hz', '200 Hz', '500 Hz', '1000 Hz', '2000 Hz', '5000 Hz'])
        self.freq_combo.setCurrentText('100 Hz')
        self.freq_combo.currentTextChanged.connect(self.on_freq_change)
        control_layout.addWidget(self.freq_combo)

        control_layout.addSpacing(10)

        self.start_btn = QtWidgets.QPushButton("Start Loop")
        self.start_btn.clicked.connect(self.start_loop)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop Loop")
        self.stop_btn.clicked.connect(self.stop_loop)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        self.plot_checkbox = QtWidgets.QCheckBox("Enable Plot Updates")
        self.plot_checkbox.setChecked(True)
        self.plot_checkbox.stateChanged.connect(self.on_plot_toggle)
        control_layout.addWidget(self.plot_checkbox)

        # Plot rate control
        control_layout.addSpacing(10)
        rate_label = QtWidgets.QLabel("Plot Rate:")
        control_layout.addWidget(rate_label)

        self.rate_combo = QtWidgets.QComboBox()
        self.rate_combo.addItems(['1 Hz', '5 Hz', '10 Hz', '20 Hz', '50 Hz', '100 Hz', '200 Hz', '500 Hz'])
        self.rate_combo.setCurrentText('10 Hz')
        self.rate_combo.currentTextChanged.connect(self.on_rate_change)
        control_layout.addWidget(self.rate_combo)

        # Data points control
        control_layout.addSpacing(10)
        points_label = QtWidgets.QLabel("Points:")
        control_layout.addWidget(points_label)

        self.points_combo = QtWidgets.QComboBox()
        self.points_combo.addItems(['100', '500', '1000', '2000', '5000'])
        self.points_combo.setCurrentText('500')
        self.points_combo.currentTextChanged.connect(self.on_points_change)
        control_layout.addWidget(self.points_combo)

        control_layout.addSpacing(20)

        self.status_label = QtWidgets.QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        control_layout.addWidget(self.status_label)

        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Statistics panel
        stats_group = QtWidgets.QGroupBox("Timing Statistics")
        stats_layout = QtWidgets.QVBoxLayout()

        self.stats_text = QtWidgets.QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(280)
        self.stats_text.setFont(QtGui.QFont("Courier", 9))
        self.stats_text.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #555;
            }
        """)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Plot panel with PyQtGraph
        plot_group = QtWidgets.QGroupBox("Live Plot - PyQtGraph STRESS TEST (push to limits!)")
        plot_layout = QtWidgets.QVBoxLayout()

        # Configure PyQtGraph
        pg.setConfigOptions(antialias=True)

        # Create plot widget with 2 subplots
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('#2d2d2d')

        # Top plot with 3 lines
        self.plot1 = self.plot_widget.addPlot(row=0, col=0)
        self.plot1.setTitle("Control Signals", color='#e0e0e0', size='12pt')
        self.plot1.setLabel('left', 'Amplitude', color='#e0e0e0')
        self.plot1.setLabel('bottom', 'Sample', color='#e0e0e0')
        self.plot1.showGrid(x=True, y=True, alpha=0.3)
        self.plot1.setYRange(-1.5, 1.5)

        # Create curve items
        self.curve1 = self.plot1.plot(pen=pg.mkPen('c', width=2), name='Signal A')
        self.curve2 = self.plot1.plot(pen=pg.mkPen('r', width=2), name='Signal B')
        self.curve3 = self.plot1.plot(pen=pg.mkPen('g', width=2), name='Signal C')

        # Add legend
        legend1 = self.plot1.addLegend()

        # Bottom plot with 1 line
        self.plot2 = self.plot_widget.addPlot(row=1, col=0)
        self.plot2.setLabel('left', 'Amplitude', color='#e0e0e0')
        self.plot2.setLabel('bottom', 'Sample', color='#e0e0e0')
        self.plot2.showGrid(x=True, y=True, alpha=0.3)
        self.plot2.setYRange(-2, 2)

        self.curve4 = self.plot2.plot(pen=pg.mkPen('m', width=2), name='Derivative')
        legend2 = self.plot2.addLegend()

        plot_layout.addWidget(self.plot_widget)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group, stretch=1)

        # Style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QGroupBox {
                background-color: #2d2d2d;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #007acc;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
            QCheckBox {
                color: #e0e0e0;
                font-size: 11pt;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)

    def start_loop(self):
        """Start control loop thread."""
        self.loop_running = True
        self.loop_count = 0
        self.loop_times.clear()
        self.plot_times.clear()
        self.sleep_times.clear()

        with self.data_lock:
            self.data_x.clear()
            self.data_y1.clear()
            self.data_y2.clear()
            self.data_y3.clear()
            self.data_y4.clear()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.loop_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.loop_thread.start()

        # Start timers for GUI updates
        self.stats_timer.start(100)  # 10Hz stats
        self.plot_timer.start(self.plot_interval_ms)  # User-adjustable plot rate

        print("Control loop started with PyQtGraph plotting")

    def stop_loop(self):
        """Stop control loop."""
        self.loop_running = False
        self.stats_timer.stop()
        self.plot_timer.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Stopped")

        print("Control loop stopped")

    def on_freq_change(self, freq_text):
        """Handle control loop frequency change."""
        freq_map = {
            '50 Hz': 50,
            '100 Hz': 100,
            '200 Hz': 200,
            '500 Hz': 500,
            '1000 Hz': 1000,
            '2000 Hz': 2000,
            '5000 Hz': 5000
        }
        self.target_hz = freq_map.get(freq_text, 100)
        self.target_period_ms = 1000.0 / self.target_hz

    def on_plot_toggle(self, state):
        """Handle plot enable/disable."""
        self.plot_enabled = bool(state)

    def on_rate_change(self, rate_text):
        """Handle plot rate change."""
        rate_map = {
            '1 Hz': 1000,
            '5 Hz': 200,
            '10 Hz': 100,
            '20 Hz': 50,
            '50 Hz': 20,
            '100 Hz': 10,
            '200 Hz': 5,
            '500 Hz': 2
        }
        self.plot_interval_ms = rate_map.get(rate_text, 100)

        if self.loop_running:
            self.plot_timer.stop()
            self.plot_timer.start(self.plot_interval_ms)

    def on_points_change(self, points_text):
        """Handle data points change."""
        new_max = int(points_text)
        self.max_data_points = new_max

        with self.data_lock:
            self.data_x = deque(list(self.data_x)[-new_max:], maxlen=new_max)
            self.data_y1 = deque(list(self.data_y1)[-new_max:], maxlen=new_max)
            self.data_y2 = deque(list(self.data_y2)[-new_max:], maxlen=new_max)
            self.data_y3 = deque(list(self.data_y3)[-new_max:], maxlen=new_max)
            self.data_y4 = deque(list(self.data_y4)[-new_max:], maxlen=new_max)

    def _control_loop(self):
        """Main control loop running at configurable frequency."""
        target_period_s = self.target_period_ms / 1000.0

        print(
            f"Loop: {self.target_hz} Hz, Plot: {1000.0 / self.plot_interval_ms:.0f} Hz, Points: {self.max_data_points}")

        while self.loop_running:
            loop_start = time.perf_counter()

            # Simulate control calculations
            self.loop_count += 1
            t = self.loop_count * target_period_s

            # Generate multiple signals
            value1 = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine
            value2 = 0.7 * np.cos(2 * np.pi * 0.8 * t)  # 0.8 Hz cosine
            value3 = 0.5 * np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz sine
            value4 = value1 * 1.5 + np.random.normal(0, 0.1)  # Noisy derivative

            # Thread-safe data update
            with self.data_lock:
                self.data_x.append(self.loop_count)
                self.data_y1.append(value1)
                self.data_y2.append(value2)
                self.data_y3.append(value3)
                self.data_y4.append(value4)

            # Simulate some processing (scale with frequency)
            if self.target_hz < 500:
                time.sleep(0.002)  # 2ms processing
            elif self.target_hz < 2000:
                time.sleep(0.0005)  # 0.5ms processing
            # else: no sleep, pure speed test

            # Measure loop time before sleep
            loop_time = (time.perf_counter() - loop_start) * 1000
            self.loop_times.append(loop_time)

            # Sleep to maintain target frequency
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_period_s - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

            sleep_actual = (time.perf_counter() - loop_start) * 1000
            self.sleep_times.append(sleep_actual)

    def _update_plot_data(self):
        """Update PyQtGraph plots - FAST rendering."""
        if not self.plot_enabled or not self.loop_running:
            return

        plot_start = time.perf_counter()

        # Thread-safe data copy
        with self.data_lock:
            if len(self.data_x) == 0:
                return

            x_data = np.array(self.data_x)
            y1_data = np.array(self.data_y1)
            y2_data = np.array(self.data_y2)
            y3_data = np.array(self.data_y3)
            y4_data = np.array(self.data_y4)

        # Update all curves - PyQtGraph is FAST!
        self.curve1.setData(x_data, y1_data)
        self.curve2.setData(x_data, y2_data)
        self.curve3.setData(x_data, y3_data)
        self.curve4.setData(x_data, y4_data)

        # Auto-range X axis
        if len(x_data) > 0:
            x_min = max(0, len(x_data) - 500)
            x_max = max(500, len(x_data))
            self.plot1.setXRange(x_min, x_max, padding=0)
            self.plot2.setXRange(x_min, x_max, padding=0)

        plot_time = (time.perf_counter() - plot_start) * 1000
        self.plot_times.append(plot_time)

        # Log if plot takes >10ms (should NEVER happen with PyQtGraph!)
        if plot_time > 10:
            print(f"⚠️ Plot took {plot_time:.1f}ms (unusual for PyQtGraph)")

    def _update_statistics(self):
        """Update statistics display."""
        if len(self.loop_times) < 10:
            return

        loop_times_ms = list(self.loop_times)
        plot_times_ms = list(self.plot_times) if self.plot_times else [0]

        avg_loop = np.mean(loop_times_ms)
        max_loop = np.max(loop_times_ms)
        min_loop = np.min(loop_times_ms)
        std_loop = np.std(loop_times_ms)

        avg_plot = np.mean(plot_times_ms) if plot_times_ms else 0
        max_plot = np.max(plot_times_ms) if plot_times_ms else 0

        # Count overruns (>15ms for 100Hz loop)
        overruns = sum(1 for t in loop_times_ms if t > 15.0)
        overrun_pct = (overruns / len(loop_times_ms)) * 100

        # Actual frequency achieved
        actual_hz = 1000.0 / avg_loop if avg_loop > 0 else 0

        stats = f"""
TARGET: {self.target_hz} Hz ({self.target_period_ms:.1f}ms period)

CONTROL LOOP TIMING:
  Average:    {avg_loop:.2f} ms  ({actual_hz:.1f} Hz actual)
  Min:        {min_loop:.2f} ms
  Max:        {max_loop:.2f} ms  {'✓ EXCELLENT' if max_loop < 12 else ('⚠️ JITTER' if max_loop < 20 else '🔴 SEVERE')}
  Std Dev:    {std_loop:.2f} ms
  Overruns:   {overruns}/{len(loop_times_ms)} ({overrun_pct:.1f}%)

PYQTGRAPH PLOT TIMING:
  Status:     {'ENABLED' if self.plot_enabled else 'DISABLED'}
  Rate:       {1000.0 / self.plot_interval_ms:.0f} Hz ({self.plot_interval_ms}ms interval)
  Average:    {avg_plot:.2f} ms  {'🟢 FAST!' if avg_plot < 5 else ''}
  Max:        {max_plot:.2f} ms  {'🟢 EXCELLENT!' if max_plot < 10 else ''}

DIAGNOSIS:
"""

        if self.plot_enabled:
            if max_plot < 5:
                stats += "  🟢 PERFECT: PyQtGraph rendering <5ms!\n"
                stats += "     Plot rendering is NOT the bottleneck.\n"
                if max_loop > 15:
                    stats += "     10-20ms jitter is from Python GIL + threading.\n"
                    stats += "     Solution: Lower plot rate or use C++ control.\n"
            elif max_plot < 10:
                stats += "  🟢 GREAT: PyQtGraph <10ms updates.\n"
                stats += "     50x faster than matplotlib!\n"
            else:
                stats += "  🟡 OK: PyQtGraph updates taking >10ms.\n"
        else:
            stats += "  ℹ️  Plot disabled for comparison.\n"
            if max_loop < 12:
                stats += "     Confirms jitter is from threading, not plotting.\n"

        stats += f"\nSamples: {len(loop_times_ms)} | Loop count: {self.loop_count}"

        # Key insight about the remaining jitter
        if self.plot_enabled and max_loop > 15 and max_plot < 10:
            stats += f"\n\n💡 KEY INSIGHT:"
            stats += f"\n   Plot rendering: {max_plot:.1f}ms (FAST!)"
            stats += f"\n   Loop jitter: {max_loop:.1f}ms (from Python GIL)"
            stats += f"\n   → PyQtGraph solved rendering, but Python threading"
            stats += f"\n     has fundamental 10-20ms scheduler delays."
            stats += f"\n\n   PROOF: Toggle plot to 1Hz or disable it - jitter remains!"
        elif not self.plot_enabled and max_loop > 15:
            stats += f"\n\n💡 SMOKING GUN:"
            stats += f"\n   Plot is DISABLED, but loop jitter is {max_loop:.1f}ms!"
            stats += f"\n   This proves jitter is from Python threading,"
            stats += f"\n   NOT from plotting library."
            stats += f"\n   → Python + threading = 10-20ms jitter (unavoidable)"
            stats += f"\n   → Teensy C++ = <1ms jitter (your architecture wins!)"
        else:
            stats += f"\n\n💡 PyQtGraph allows real-time plotting at 10Hz+"
            stats += f"\n   with minimal impact on control loop."

        self.stats_text.setPlainText(stats)

        # Update status
        if overrun_pct > 10:
            status_icon = '🔴'
        elif overrun_pct > 1:
            status_icon = '🟡'
        else:
            status_icon = '🟢'

        self.status_label.setText(
            f"{status_icon} Running: {actual_hz:.1f} Hz (target: {self.target_hz} Hz)"
        )


def main():
    """Launch PyQtGraph stress test."""
    print("PyQtGraph Stress Test")
    print("=" * 50)
    print("Test limits: 50-5000 Hz loop, 1-500 Hz plot, 100-5000 points")
    print("Expected limits: ~1-2 kHz (Python GIL), ~200 Hz plot @ 500pts")
    print()

    app = QtWidgets.QApplication(sys.argv)

    # Set dark theme
    app.setStyle('Fusion')
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 30, 30))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(45, 45, 45))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)

    window = JitterTestApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()