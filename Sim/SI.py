#!/usr/bin/env python3
"""
Hardware Step Response Data Collector

Sends step inputs to the Stewart platform and records:
- Ball position from camera
- Commanded servo angles
- Platform angles (rx, ry)

Usage:
    python collect_step_response.py <serial_port>

Example:
    python collect_step_response.py COM3
"""

import sys
import time
import csv
import numpy as np
from datetime import datetime

from setup.hardware_controller_config import SerialController
from core.core import StewartPlatformIK


class StepResponseCollector:
    """Collect step response data from hardware."""

    def __init__(self, serial_port, output_file=None):
        self.serial_port = serial_port

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"step_response_{timestamp}.csv"

        self.output_file = output_file

        # Initialize IK
        self.ik = StewartPlatformIK(
            horn_length=31.75,
            rod_length=145.0,
            base=73.025,
            base_anchors=36.8893,
            platform=67.775,
            platform_anchors=12.7,
            top_surface_offset=26.0
        )

        # Camera parameters
        self.pixy_width_mm = 350.0
        self.pixy_height_mm = 266.0
        self.pixels_to_mm_x = self.pixy_width_mm / 316.0
        self.pixels_to_mm_y = self.pixy_height_mm / 208.0

        # Data storage
        self.data_records = []

        print(f"Output file: {self.output_file}")

    def connect(self):
        """Connect to hardware."""
        print(f"\nConnecting to {self.serial_port}...")
        self.serial = SerialController(self.serial_port)
        success, message = self.serial.connect()

        if not success:
            raise RuntimeError(f"Connection failed: {message}")

        print("Connected successfully")

        # Configure servos for fast response
        time.sleep(0.5)
        self.serial.set_servo_speed(0)  # Unlimited speed
        time.sleep(0.1)
        self.serial.set_servo_acceleration(0)  # No ramping
        time.sleep(0.2)
        print("Servos configured: Speed=0 (unlimited), Accel=0")

    def calculate_angles(self, rx, ry):
        """Calculate servo angles for given tilt."""
        translation = np.array([0.0, 0.0, self.ik.home_height_top_surface])
        rotation = np.array([rx, ry, 0.0])

        angles = self.ik.calculate_servo_angles(translation, rotation, use_top_surface_offset=True)

        if angles is None:
            raise ValueError(f"IK solution failed for rx={rx}, ry={ry}")

        return angles

    def send_angles_and_wait(self, angles, settle_time=0.5):
        """Send angles to hardware and wait for settling."""
        success = self.serial.send_servo_angles(angles)
        if not success:
            print("Warning: Failed to send angles")
        time.sleep(settle_time)

    def get_ball_position(self):
        """Get current ball position from camera."""
        # Clear old data
        while not self.serial.ball_data_queue.empty():
            self.serial.ball_data_queue.get_nowait()

        # Wait for fresh data
        time.sleep(0.05)

        ball_data = self.serial.get_latest_ball_data()

        if ball_data is None:
            return None, None, False

        # Convert to mm with Y-axis inversion
        CAMERA_HEIGHT_PIXELS = 208.0
        CAMERA_CENTER_X = 158.0
        CAMERA_CENTER_Y = 104.0

        ball_x_mm = (ball_data['x'] - CAMERA_CENTER_X) * self.pixels_to_mm_x
        ball_y_mm = ((CAMERA_HEIGHT_PIXELS - ball_data['y']) - CAMERA_CENTER_Y) * self.pixels_to_mm_y

        return ball_x_mm, ball_y_mm, ball_data['detected']

    def record_data_point(self, timestamp, rx, ry, angles, ball_x, ball_y, detected):
        """Record a single data point."""
        record = {
            'time': timestamp,
            'rx_deg': rx,
            'ry_deg': ry,
            'ball_x_mm': ball_x if detected else np.nan,
            'ball_y_mm': ball_y if detected else np.nan,
            'ball_detected': detected,
            's0': angles[0],
            's1': angles[1],
            's2': angles[2],
            's3': angles[3],
            's4': angles[4],
            's5': angles[5],
        }
        self.data_records.append(record)

    def run_step_test(self, rx_step, ry_step, duration=5.0, sample_rate=50.0):
        """
        Run a step response test.

        Args:
            rx_step: Step size in rx direction (degrees)
            ry_step: Step size in ry direction (degrees)
            duration: Test duration AFTER step (seconds)
            sample_rate: Data collection rate (Hz)
        """
        print(f"\n{'=' * 60}")
        print(f"Step Test: rx={rx_step}°, ry={ry_step}°")
        print(f"Pre-step: 1s, Post-step: {duration}s, Sample rate: {sample_rate}Hz")
        print(f"{'=' * 60}")

        # Calculate angles for step and neutral
        step_angles = self.calculate_angles(rx_step, ry_step)
        neutral_angles = self.calculate_angles(0.0, 0.0)

        print(f"Step servo angles: {step_angles}")

        # Ensure we're at neutral
        print("Setting neutral position...")
        self.send_angles_and_wait(neutral_angles, settle_time=0.5)

        # Record 1 second of baseline at neutral
        print("Recording baseline (1s at neutral)...")
        start_time = time.time()
        sample_interval = 1.0 / sample_rate
        next_sample = start_time + sample_interval
        samples_collected = 0
        step_applied = False
        step_time = start_time + 1.0  # Apply step after 1 second

        total_duration = 1.0 + duration  # 1s pre + duration post

        while time.time() - start_time < total_duration:
            current_time = time.time()

            # Apply step input at t=1.0s
            if not step_applied and current_time >= step_time:
                print("Applying step input NOW!")
                success = self.serial.send_servo_angles(step_angles)
                if not success:
                    print("Warning: Failed to send step angles")
                step_applied = True

            if current_time >= next_sample:
                timestamp = current_time - start_time

                # Get ball position
                ball_x, ball_y, detected = self.get_ball_position()

                # Record with current platform state (neutral before step, step after)
                if step_applied:
                    self.record_data_point(timestamp, rx_step, ry_step, step_angles, ball_x, ball_y, detected)
                else:
                    self.record_data_point(timestamp, 0.0, 0.0, neutral_angles, ball_x, ball_y, detected)

                samples_collected += 1
                if samples_collected % 10 == 0:
                    status = "detected" if detected else "NOT detected"
                    phase = "PRE-STEP" if not step_applied else "POST-STEP"
                    print(f"  [{phase}] t={timestamp:.2f}s: ball {status}")

                next_sample += sample_interval

            time.sleep(0.001)  # Small sleep to prevent busy waiting

        print(f"Collected {samples_collected} samples (baseline + step response)")

    def return_to_neutral(self):
        """Return platform to neutral position."""
        print("\nReturning to neutral...")
        neutral_angles = self.calculate_angles(0.0, 0.0)
        self.send_angles_and_wait(neutral_angles, settle_time=1.0)
        print("Neutral position reached")

    def save_data(self):
        """Save collected data to CSV."""
        if not self.data_records:
            print("No data to save")
            return

        print(f"\nSaving {len(self.data_records)} records to {self.output_file}...")

        fieldnames = ['time', 'rx_deg', 'ry_deg', 'ball_x_mm', 'ball_y_mm', 'ball_detected',
                      's0', 's1', 's2', 's3', 's4', 's5']

        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data_records)

        print(f"Data saved successfully to {self.output_file}")

    def run_interactive_tests(self):
        """Run tests interactively with manual ball resets."""
        print("\n" + "=" * 60)
        print("INTERACTIVE STEP RESPONSE TEST")
        print("=" * 60)
        print("\nOptions:")
        print("  - Enter rx and ry angles (e.g., '15 0' for +15° rx)")
        print("  - Type '0' to return to neutral")
        print("  - Type 'q' to save and quit")

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

                if abs(rx) > 15 or abs(ry) > 15:
                    print("Error: Angles must be within ±15°")
                    continue

                description = f"rx={rx}°, ry={ry}°"

                print(f"\n>>> Test: {description}")
                print(">>> PLACE BALL AT CENTER <<<")
                input("Press ENTER when ball is ready...")

                print(f"Starting test in 3 seconds...")
                for i in range(3, 0, -1):
                    print(f"  {i}...")
                    time.sleep(1.0)

                print("\nTest sequence:")
                print("  1. Record 1s baseline at neutral (0°, 0°)")
                print(f"  2. Apply step to ({rx}°, {ry}°)")
                print("  3. Record 5s response")

                self.run_step_test(rx, ry, duration=50.0, sample_rate=50.0)

                print("\nTest complete!")
                self.return_to_neutral()

            except ValueError:
                print("Invalid format. Enter two numbers (e.g., '15 0' or '-10 5')")

        print("\n" + "=" * 60)
        print("TESTING SESSION COMPLETE")
        print("=" * 60)

    def disconnect(self):
        """Disconnect from hardware."""
        if hasattr(self, 'serial'):
            self.serial.disconnect()
            print("\nDisconnected from hardware")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python collect_step_response.py <serial_port>")
        print("\nExample:")
        print("  python collect_step_response.py COM3")
        print("  python collect_step_response.py /dev/ttyACM0")
        sys.exit(1)

    serial_port = sys.argv[1]

    collector = StepResponseCollector(serial_port)

    try:
        collector.connect()
        collector.run_interactive_tests()
        collector.save_data()

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        collector.return_to_neutral()
        collector.save_data()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        collector.disconnect()


if __name__ == "__main__":
    main()