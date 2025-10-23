#!/usr/bin/env python3
"""
Matplotlib Jitter Test - AGGRESSIVE MODE

Demonstrates severe timing jitter caused by matplotlib plot updates in real-time
control loops. This version uses aggressive rendering to guarantee 50ms+ delays.

Features:
- 100Hz control loop (10ms target period)
- 50Hz plot updates (20ms interval - aggressive!)
- 2 subplots with 4 lines (500 points each)
- Synchronous rendering (blocks until complete)
- Forced layout recalculation

Usage:
    python matplotlib_jitter_test.py

Press 'Start Loop' then toggle 'Enable Plot Updates' to see the dramatic difference.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import deque


class JitterTestApp:
    """Test application for matplotlib jitter analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("Matplotlib Jitter Test - 100Hz Control Loop")
        self.root.geometry("900x800")

        # Control loop parameters
        self.target_hz = 100
        self.target_period_ms = 1000.0 / self.target_hz
        self.loop_running = False
        self.plot_enabled = tk.BooleanVar(value=True)

        # Timing statistics
        self.loop_times = deque(maxlen=1000)
        self.plot_times = deque(maxlen=1000)
        self.sleep_times = deque(maxlen=1000)
        self.loop_count = 0

        # Simulated data for multiple signals
        self.data_x = deque(maxlen=500)
        self.data_y1 = deque(maxlen=500)
        self.data_y2 = deque(maxlen=500)
        self.data_y3 = deque(maxlen=500)
        self.data_y4 = deque(maxlen=500)

        self.setup_gui()
        self.setup_plot()

    def setup_gui(self):
        """Create GUI controls."""
        # Control panel
        control_frame = ttk.LabelFrame(self.root, text="Control", padding=10)
        control_frame.pack(side='top', fill='x', padx=10, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Start Loop",
                                    command=self.start_loop, width=12)
        self.start_btn.pack(side='left', padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop Loop",
                                   command=self.stop_loop, state='disabled', width=12)
        self.stop_btn.pack(side='left', padx=5)

        ttk.Checkbutton(control_frame, text="Enable Plot Updates",
                        variable=self.plot_enabled,
                        command=self.on_plot_toggle).pack(side='left', padx=20)

        self.status_label = ttk.Label(control_frame, text="Status: Idle",
                                      font=('Consolas', 10, 'bold'))
        self.status_label.pack(side='left', padx=20)

        # Statistics panel
        stats_frame = ttk.LabelFrame(self.root, text="Timing Statistics", padding=10)
        stats_frame.pack(side='top', fill='x', padx=10, pady=5)

        self.stats_text = tk.Text(stats_frame, height=12, font=('Consolas', 9),
                                  bg='#2d2d2d', fg='#e0e0e0')
        self.stats_text.pack(fill='both', expand=True)

    def setup_plot(self):
        """Create matplotlib plot with multiple complex elements."""
        plot_frame = ttk.LabelFrame(self.root, text="Live Plot - AGGRESSIVE MODE (causes 50ms+ jitter)", padding=10)
        plot_frame.pack(side='top', fill='both', expand=True, padx=10, pady=5)

        plt.style.use('dark_background')
        # Multiple subplots to increase rendering load
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 5), facecolor='#2d2d2d')
        self.ax1.set_facecolor('#3d3d3d')
        self.ax2.set_facecolor('#3d3d3d')

        self.ax1.set_xlim(0, 500)
        self.ax1.set_ylim(-1.5, 1.5)
        self.ax1.set_xlabel('Sample')
        self.ax1.set_ylabel('Signal 1')
        self.ax1.set_title('Control Data (Multiple Lines)')
        self.ax1.grid(True, alpha=0.3)

        self.ax2.set_xlim(0, 500)
        self.ax2.set_ylim(-2, 2)
        self.ax2.set_xlabel('Sample')
        self.ax2.set_ylabel('Signal 2')
        self.ax2.grid(True, alpha=0.3)

        # Multiple lines to increase rendering complexity
        self.line1, = self.ax1.plot([], [], 'c-', linewidth=2, label='Signal A')
        self.line2, = self.ax1.plot([], [], 'r-', linewidth=2, label='Signal B', alpha=0.7)
        self.line3, = self.ax1.plot([], [], 'g-', linewidth=2, label='Signal C', alpha=0.7)
        self.line4, = self.ax2.plot([], [], 'm-', linewidth=2, label='Derivative')

        # Legends force re-layout on every update
        self.ax1.legend(loc='upper right')
        self.ax2.legend(loc='upper right')

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.draw()

    def start_loop(self):
        """Start control loop thread."""
        self.loop_running = True
        self.loop_count = 0
        self.loop_times.clear()
        self.plot_times.clear()
        self.sleep_times.clear()
        self.data_x.clear()
        self.data_y1.clear()
        self.data_y2.clear()
        self.data_y3.clear()
        self.data_y4.clear()

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')

        self.loop_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.loop_thread.start()

        # Start GUI update loop at HIGH FREQUENCY to cause jitter
        self._gui_update_loop()

    def stop_loop(self):
        """Stop control loop."""
        self.loop_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.status_label.config(text="Status: Stopped")

    def on_plot_toggle(self):
        """Handle plot enable/disable."""
        enabled = self.plot_enabled.get()
        status = "ENABLED (expect jitter!)" if enabled else "DISABLED (smooth)"
        print(f"Plot updates: {status}")

    def _control_loop(self):
        """Main control loop running at 100Hz."""
        target_period_s = self.target_period_ms / 1000.0

        while self.loop_running:
            loop_start = time.perf_counter()

            # Simulate control calculations
            self.loop_count += 1
            t = self.loop_count * 0.01  # 10ms timestep

            # Generate multiple signals
            value1 = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine
            value2 = 0.7 * np.cos(2 * np.pi * 0.8 * t)  # 0.8 Hz cosine
            value3 = 0.5 * np.sin(2 * np.pi * 1.2 * t)  # 1.2 Hz sine
            value4 = value1 * 1.5 + np.random.normal(0, 0.1)  # Noisy derivative

            self.data_x.append(self.loop_count)
            self.data_y1.append(value1)
            self.data_y2.append(value2)
            self.data_y3.append(value3)
            self.data_y4.append(value4)

            # Simulate some processing (2ms)
            time.sleep(0.002)

            # Measure loop time before sleep
            loop_time = (time.perf_counter() - loop_start) * 1000
            self.loop_times.append(loop_time)

            # Sleep to maintain 100Hz
            sleep_start = time.perf_counter()
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_period_s - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

            sleep_actual = (time.perf_counter() - sleep_start) * 1000
            self.sleep_times.append(sleep_actual)

            # Check for overruns
            total_time = (time.perf_counter() - loop_start) * 1000
            if total_time > self.target_period_ms * 1.5:
                print(f"WARNING: Loop overrun! {total_time:.1f}ms (target: {self.target_period_ms:.1f}ms)")

    def _gui_update_loop(self):
        """GUI update loop - AGGRESSIVE updates to cause severe jitter."""
        if not self.loop_running:
            return

        update_start = time.perf_counter()

        # Update statistics
        self._update_statistics()

        # Update plot with AGGRESSIVE rendering (THIS CAUSES SEVERE JITTER!)
        if self.plot_enabled.get():
            plot_start = time.perf_counter()

            if len(self.data_x) > 0:
                # Update all 4 lines with full dataset
                x_data = list(self.data_x)
                y1_data = list(self.data_y1)
                y2_data = list(self.data_y2)
                y3_data = list(self.data_y3)
                y4_data = list(self.data_y4)

                self.line1.set_data(x_data, y1_data)
                self.line2.set_data(x_data, y2_data)
                self.line3.set_data(x_data, y3_data)
                self.line4.set_data(x_data, y4_data)

                # Update axis limits
                if len(x_data) > 0:
                    x_max = max(500, len(x_data))
                    x_min = max(0, len(x_data) - 500)
                    self.ax1.set_xlim(x_min, x_max)
                    self.ax2.set_xlim(x_min, x_max)

                # Force layout recalculation (adds to rendering time)
                self.fig.tight_layout()

                # CRITICAL: Use synchronous draw() instead of draw_idle()
                # This BLOCKS until rendering completes (50-200ms!)
                self.canvas.draw()
                self.canvas.flush_events()

            plot_time = (time.perf_counter() - plot_start) * 1000
            self.plot_times.append(plot_time)

            if plot_time > 50:
                print(f"🔴 SEVERE JITTER: Plot took {plot_time:.1f}ms!")

        # Schedule next update at HIGH FREQUENCY (20ms = 50Hz)
        # This ensures plot updates compete heavily with control loop
        self.root.after(20, self._gui_update_loop)

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
  Max:        {max_loop:.2f} ms  {'⚠️ JITTER!' if max_loop > 20 else '✓ OK'}
  Std Dev:    {std_loop:.2f} ms
  Overruns:   {overruns}/{len(loop_times_ms)} ({overrun_pct:.1f}%)

PLOT UPDATE TIMING:
  Status:     {'ENABLED' if self.plot_enabled.get() else 'DISABLED'}
  Average:    {avg_plot:.2f} ms
  Max:        {max_plot:.2f} ms  {'⚠️ CAUSES DELAYS!' if max_plot > 30 else ''}

DIAGNOSIS:
"""

        if self.plot_enabled.get():
            if max_plot > 50:
                stats += "  🔴 SEVERE: Plot updates causing >50ms delays!\n"
                stats += "     This will destroy real-time control performance.\n"
            elif max_plot > 20:
                stats += "  🟡 WARNING: Plot updates causing 20-50ms delays.\n"
                stats += "     Control loop jitter detected.\n"
            else:
                stats += "  🟢 OK: Plot updates under control (for now).\n"
        else:
            stats += "  🟢 OPTIMAL: Plot disabled, smooth control loop.\n"

        stats += f"\nSamples: {len(loop_times_ms)} | Loop count: {self.loop_count}"

        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', stats)

        # Update status
        status_color = '🔴' if overrun_pct > 10 else ('🟡' if overrun_pct > 1 else '🟢')
        self.status_label.config(
            text=f"{status_color} Running: {actual_hz:.1f} Hz (target: {self.target_hz} Hz)"
        )


def main():
    """Launch jitter test application."""
    print("=" * 70)
    print("MATPLOTLIB JITTER TEST - AGGRESSIVE MODE")
    print("=" * 70)
    print()
    print("This demonstrates how matplotlib plot updates cause SEVERE timing")
    print("jitter in real-time control loops.")
    print()
    print("AGGRESSIVE MODE FEATURES:")
    print("  - 2 subplots with 4 lines (500 points each)")
    print("  - Synchronous rendering (blocks until complete)")
    print("  - 50Hz plot updates (competes with 100Hz control)")
    print("  - Forced layout recalculation every frame")
    print()
    print("INSTRUCTIONS:")
    print("  1. Click 'Start Loop' to begin 100Hz control loop")
    print("  2. Observe SEVERE jitter with plot ENABLED")
    print("  3. Toggle 'Enable Plot Updates' OFF")
    print("  4. Watch jitter disappear completely!")
    print()
    print("EXPECTED BEHAVIOR:")
    print("  - Plot ENABLED:  Max loop time 50-200ms (SEVERE JITTER!)")
    print("  - Plot DISABLED: Max loop time <5ms (PERFECTLY SMOOTH)")
    print()
    print("=" * 70)
    print()

    root = tk.Tk()
    app = JitterTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()