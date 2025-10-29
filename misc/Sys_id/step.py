#!/usr/bin/env python3
"""
Hardware Step Response Data Collector - Continuous Logging

Sends step inputs to the Stewart platform and records ball position data.
Teensy continuously prints BALL: data, Python continuously logs to CSV.

Usage:
    python step.py <serial_port>

Example:
    python step.py COM3
"""

import sys
import time
import csv
import numpy as np
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
import threading
import serial

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.core import StewartPlatformIK


class StepResponseCollector:
    """Collect step response data from hardware with continuous logging."""

    def __init__(self, serial_port, output_file=None):
        self.serial_port_name = serial_port

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"step_response_{timestamp}.csv"

        self.output_file = output_file

        # Initialize IK
        self.ik = StewartPlatformIK(
            horn_length=45.3722,
            rod_length=205.0,
            base=86.6025 + 18.75 + 11,
            base_anchors=64.75,
            platform=84.0759,
            platform_anchors=12.5,
            top_surface_offset=38.0
        )

        # Camera parameters (for reference only - Teensy sends pixels)
        self.pixy_width_mm = 350.0  # Field of view width
        self.pixy_height_mm = 266.0  # Field of view height
        self.pixy_width_px = 316.0  # Pixy2 resolution
        self.pixy_height_px = 208.0

        # Serial connection
        self.serial = None
        self.running = False
        self.read_thread = None

        # Data queue for continuous data capture
        self.ball_data_queue = Queue(maxsize=5000)

        # Current platform state
        self.current_rx = 0.0
        self.current_ry = 0.0
        self.current_angles = np.zeros(6)

        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        self.samples_written = 0

        print(f"Output file: {self.output_file}")

    def connect(self):
        """Connect to hardware and start background read thread."""
        print(f"\nConnecting to {self.serial_port_name}...")

        try:
            self.serial = serial.Serial(self.serial_port_name, baudrate=200000, timeout=0.1)
            time.sleep(2.0)

            # Clear startup messages
            self.serial.reset_input_buffer()

            print("Connected successfully")

            # Start background read thread
            self.running = True
            self.read_thread = threading.Thread(target=self._serial_read_loop, daemon=True)
            self.read_thread.start()
            print("Background read thread started")

            # Set servo speed to unlimited
            time.sleep(0.5)
            self.serial.write(b"SPD:0\n")
            time.sleep(0.1)
            print("Servos configured: Speed=0 (unlimited)")

        except Exception as e:
            raise RuntimeError(f"Connection failed: {e}")

    def _serial_read_loop(self):
        """Background thread to continuously read serial data."""
        buffer = ""

        while self.running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting > 0:
                    chunk = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='ignore')
                    buffer += chunk

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if line.startswith("BALL:"):
                            # Parse: BALL:timestamp_s,x_px,y_px,detected,error_x_px,error_y_px
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

                                    # Add current platform state
                                    ball_data['rx'] = self.current_rx
                                    ball_data['ry'] = self.current_ry
                                    ball_data['servo_angles'] = self.current_angles.copy()
                                    ball_data['pc_time'] = time.time()

                                    if not self.ball_data_queue.full():
                                        self.ball_data_queue.put(ball_data)
                            except (ValueError, IndexError):
                                pass

                time.sleep(0.0005)

            except Exception as e:
                if self.running:
                    print(f"Serial read error: {e}")
                time.sleep(0.1)

    def calculate_angles(self, rx, ry):
        """Calculate servo angles for given tilt."""
        translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
        rotation = np.array([rx, ry, 0.0])

        angles = self.ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

        if angles is None:
            raise ValueError(f"IK solution failed for rx={rx}, ry={ry}")

        return angles

    def send_angles(self, angles):
        """Send angles to hardware."""
        if not self.serial or not self.serial.is_open:
            print("Warning: Serial not connected")
            return False

        command = ",".join([f"{angle:.3f}" for angle in angles]) + "\n"

        try:
            self.serial.write(command.encode('utf-8'))
            self.serial.flush()
            return True
        except Exception as e:
            print(f"Warning: Failed to send angles: {e}")
            return False

    def set_position(self, rx, ry):
        """Calculate and send position, update internal state."""
        angles = self.calculate_angles(rx, ry)
        success = self.send_angles(angles)

        if success:
            self.current_rx = rx
            self.current_ry = ry
            self.current_angles = angles

        return success

    def start_logging(self):
        """Open CSV file and start logging."""
        print(f"\nOpening CSV file: {self.output_file}")

        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # Write header
        self.csv_writer.writerow([
            'pc_time', 'teensy_time_s',
            'ball_x_px', 'ball_y_px', 'ball_detected',
            'ball_error_x_px', 'ball_error_y_px',
            'rx_deg', 'ry_deg',
            's0', 's1', 's2', 's3', 's4', 's5'
        ])

        self.samples_written = 0
        print("CSV logging started")

    def stop_logging(self):
        """Close CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"\nCSV logging stopped. Total samples: {self.samples_written}")

    def process_data_queue(self):
        """Process queued ball data and write to CSV."""
        samples_processed = 0

        while not self.ball_data_queue.empty():
            try:
                data = self.ball_data_queue.get_nowait()

                if self.csv_writer:
                    # Write row: pc_time, teensy_time_s, ball x/y/detected/errors (pixels), platform rx/ry, servo angles
                    self.csv_writer.writerow([
                        data['pc_time'],
                        data['timestamp'],
                        data['x'],
                        data['y'],
                        int(data['detected']),
                        data['error_x'],
                        data['error_y'],
                        data['rx'],
                        data['ry'],
                        data['servo_angles'][0],
                        data['servo_angles'][1],
                        data['servo_angles'][2],
                        data['servo_angles'][3],
                        data['servo_angles'][4],
                        data['servo_angles'][5]
                    ])
                    self.samples_written += 1
                    samples_processed += 1

            except Empty:
                break

        return samples_processed

    def run_step_test(self, rx_step, ry_step, baseline_duration=1.0, step_duration=5.0):
        """
        Run a step response test with continuous data logging.

        Args:
            rx_step: Step size in rx direction (degrees)
            ry_step: Step size in ry direction (degrees)
            baseline_duration: Duration at neutral before step (seconds)
            step_duration: Duration after step (seconds)
        """
        print(f"\n{'=' * 60}")
        print(f"Step Test: rx={rx_step}°, ry={ry_step}°")
        print(f"Baseline: {baseline_duration}s, Step: {step_duration}s")
        print(f"{'=' * 60}")

        # Clear queue
        while not self.ball_data_queue.empty():
            try:
                self.ball_data_queue.get_nowait()
            except Empty:
                break

        # Move to neutral
        print("Moving to neutral position...")
        self.set_position(0.0, 0.0)
        time.sleep(1.0)

        # Start logging
        self.start_logging()

        print(f"\nRecording {baseline_duration}s baseline at neutral...")
        start_time = time.time()
        step_time = start_time + baseline_duration
        step_applied = False

        total_duration = baseline_duration + step_duration
        last_report_time = start_time
        report_interval = 0.5

        while time.time() - start_time < total_duration:
            current_time = time.time()

            # Apply step at baseline_duration
            if not step_applied and current_time >= step_time:
                print(f"\nApplying step input: rx={rx_step}°, ry={ry_step}°")
                self.set_position(rx_step, ry_step)
                step_applied = True

            # Process and log data
            samples = self.process_data_queue()

            # Progress report
            if current_time - last_report_time >= report_interval:
                elapsed = current_time - start_time
                phase = "BASELINE" if not step_applied else "STEP"
                queue_size = self.ball_data_queue.qsize()
                print(f"  [{phase}] t={elapsed:.1f}s | Samples: {self.samples_written} | Queue: {queue_size}")
                last_report_time = current_time

            time.sleep(0.01)

        # Process remaining queued data
        print("\nProcessing remaining data...")
        time.sleep(0.5)
        final_samples = self.process_data_queue()

        print(f"\nTest complete! Total samples: {self.samples_written}")

        # Stop logging
        self.stop_logging()

    def return_to_neutral(self):
        """Return platform to neutral position."""
        print("\nReturning to neutral...")
        self.set_position(0.0, 0.0)
        time.sleep(1.0)
        print("Neutral position reached")

    def run_interactive_tests(self):
        """Run tests interactively with manual ball resets."""
        print("\n" + "=" * 60)
        print("INTERACTIVE STEP RESPONSE TEST")
        print("=" * 60)
        print("\nOptions:")
        print("  - Enter rx and ry angles (e.g., '15 0' for +15° rx)")
        print("  - Type '0' to return to neutral")
        print("  - Type 'q' to quit")

        while True:
            print("\n" + "-" * 60)
            choice = input("\nEnter angles 'rx ry' (or '0'/'q'): ").strip().lower()

            if choice == 'q':
                print("\nExiting test mode...")
                break

            if choice == '0':
                self.return_to_neutral()
                continue

            try:
                parts = choice.split()
                if len(parts) != 2:
                    print("Error: Enter two numbers separated by space (e.g., '15 0')")
                    continue

                rx = float(parts[0])
                ry = float(parts[1])

                if abs(rx) > 20 or abs(ry) > 20:
                    print("Error: Angles must be within ±20°")
                    continue

                print(f"\n>>> Test: rx={rx}°, ry={ry}°")
                print(">>> PLACE BALL AT CENTER <<<")
                input("Press ENTER when ball is ready...")

                print(f"Starting test in 3 seconds...")
                for i in range(3, 0, -1):
                    print(f"  {i}...")
                    time.sleep(1.0)

                self.run_step_test(rx, ry, baseline_duration=1.0, step_duration=5.0)

                print("\nTest complete!")
                self.return_to_neutral()

            except ValueError:
                print("Invalid format. Enter two numbers (e.g., '15 0' or '-10 5')")

        print("\n" + "=" * 60)
        print("TESTING SESSION COMPLETE")
        print("=" * 60)

    def disconnect(self):
        """Disconnect from hardware and stop threads."""
        print("\nDisconnecting...")

        # Stop read thread
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)

        # Close CSV if open
        if self.csv_file:
            self.stop_logging()

        # Close serial
        if self.serial and self.serial.is_open:
            self.serial.close()

        print("Disconnected")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python step.py <serial_port>")
        print("\nExample:")
        print("  python step.py COM3")
        print("  python step.py /dev/ttyACM0")
        sys.exit(1)

    serial_port = sys.argv[1]

    collector = StepResponseCollector(serial_port)

    try:
        collector.connect()
        collector.run_interactive_tests()

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        collector.return_to_neutral()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        collector.disconnect()


if __name__ == "__main__":
    main()
