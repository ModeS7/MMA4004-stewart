#!/usr/bin/env python3
"""
IMU Mirror Demo - Platform mimics IMU orientation

Platform follows the IMU tilt with zero translation.
Tests IMU Kalman filter and basic orientation control.

Usage:
    python imu_mirror_demo.py --port COM3
    python imu_mirror_demo.py --port COM3 --calibrate
"""

import serial
import serial.tools.list_ports
import numpy as np
import time
import argparse
import threading
from queue import Queue, Empty
from collections import deque

from core.control_core import IMUKalmanFilter
from core.core import StewartPlatformIK


class IMUMirrorController:
    """
    IMU mirror controller: platform mimics IMU orientation.

    Features:
    - Dual-rate Kalman filter (gyro ~759 Hz, accel ~1265 Hz)
    - Real-time servo control
    - Calibration mode
    """

    def __init__(self, port, baudrate=2000000):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False

        # IMU Kalman filter
        self.kalman = IMUKalmanFilter(
            gyro_scale=0.00875,   # L3GD20: 0.00875 deg/s per raw unit (250 dps mode)
            accel_scale=0.001,    # LSM303: 0.001 g per raw unit (2g mode)
            dt_gyro=1.0/759.0,    # Gyro sampling rate
            dynamic_accel_threshold=2.0  # m/s² threshold for quasi-static detection
        )

        # Platform inverse kinematics
        self.ik = StewartPlatformIK()

        # Communication threads
        self.read_thread = None
        self.control_thread = None
        self.running = False

        # Data queues
        self.accel_queue = Queue(maxsize=100)
        self.gyro_queue = Queue(maxsize=100)
        self.command_queue = Queue(maxsize=20)

        # Calibration
        self.calibration_mode = False
        self.calibration_samples = []
        self.calibration_duration = 3.0

        # Statistics
        self.gyro_count = 0
        self.accel_count = 0
        self.update_count = 0
        self.rejected_update_count = 0
        self.servo_command_count = 0
        self.start_time = time.time()

        # Current state
        self.current_rx = 0.0
        self.current_ry = 0.0
        self.last_accel_mag = 0.0

    def connect(self):
        """Connect to Teensy running IMU_control.ino"""
        try:
            print(f"Connecting to {self.port} at {self.baudrate} baud...")
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2.5)  # Wait for Teensy reset

            # Read startup messages
            startup_messages = []
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                if self.serial.in_waiting:
                    try:
                        line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            startup_messages.append(line)
                            # Only print non-IMU data messages
                            if not line.startswith("A:") and not line.startswith("G:"):
                                print(f"  Teensy: {line}")
                    except:
                        pass

            self.connected = True
            print("Connected successfully\n")
            return True

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from Teensy"""
        self.running = False

        if self.read_thread:
            self.read_thread.join(timeout=1.0)
        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        if self.serial and self.serial.is_open:
            self.serial.close()

        self.connected = False
        print("\nDisconnected")

    def _read_loop(self):
        """Read IMU data from serial port"""
        while self.running:
            try:
                if self.serial.in_waiting:
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()

                    if line.startswith("A:"):
                        # Accelerometer: A:timestamp_us,ax,ay,az
                        parts = line[2:].split(',')
                        if len(parts) == 4:
                            try:
                                timestamp_us = int(parts[0])
                                ax = int(parts[1])
                                ay = int(parts[2])
                                az = int(parts[3])

                                if not self.accel_queue.full():
                                    self.accel_queue.put((timestamp_us, ax, ay, az))
                                self.accel_count += 1

                            except ValueError:
                                pass

                    elif line.startswith("G:"):
                        # Gyroscope: G:timestamp_us,gx,gy,gz
                        parts = line[2:].split(',')
                        if len(parts) == 4:
                            try:
                                timestamp_us = int(parts[0])
                                gx = int(parts[1])
                                gy = int(parts[2])
                                gz = int(parts[3])

                                if not self.gyro_queue.full():
                                    self.gyro_queue.put((timestamp_us, gx, gy, gz))
                                self.gyro_count += 1

                            except ValueError:
                                pass

                    elif line.startswith("READY:"):
                        print(f"  {line}")
                    elif line.startswith("RATE:"):
                        pass  # Suppress rate reports from Arduino

            except Exception as e:
                if self.running:
                    print(f"Read error: {e}")
                break

    def _control_loop(self):
        """Main control loop: process IMU data and send servo commands"""
        last_status_time = time.time()
        status_interval = 1.0  # Print status every second

        # Latest accelerometer reading
        latest_accel = None

        while self.running:
            try:
                # Process all gyro samples (prediction at ~759 Hz)
                gyro_processed = False
                while not self.gyro_queue.empty():
                    timestamp_us, gx, gy, gz = self.gyro_queue.get_nowait()

                    if self.calibration_mode:
                        # Collect samples for calibration
                        self.calibration_samples.append([gx, gy, gz])
                    else:
                        # Run Kalman prediction
                        state = self.kalman.predict([gx, gy, gz])
                        self.current_rx, self.current_ry = state[0], state[1]
                        gyro_processed = True

                # Process accelerometer samples (update at ~1265 Hz)
                while not self.accel_queue.empty():
                    timestamp_us, ax, ay, az = self.accel_queue.get_nowait()
                    latest_accel = [ax, ay, az]

                # Update with latest accelerometer reading if available
                if latest_accel is not None and not self.calibration_mode:
                    updated, state = self.kalman.update(latest_accel)
                    if updated:
                        self.update_count += 1
                        self.current_rx, self.current_ry = state[0], state[1]
                    else:
                        self.rejected_update_count += 1

                    self.last_accel_mag = self.kalman.last_accel_magnitude
                    latest_accel = None  # Clear after processing

                # Send servo commands if orientation changed
                if gyro_processed and not self.calibration_mode:
                    self.send_servo_command(self.current_rx, self.current_ry)

                # Print status periodically
                if time.time() - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = time.time()

                # Small sleep to prevent CPU spinning
                time.sleep(0.0001)

            except Exception as e:
                if self.running:
                    print(f"Control loop error: {e}")
                break

    def send_servo_command(self, rx_deg, ry_deg):
        """
        Compute servo angles for given orientation and send to Teensy.

        Args:
            rx_deg: Roll angle in degrees
            ry_deg: Pitch angle in degrees
        """
        try:
            # Platform pose: match IMU orientation with zero translation
            translation = np.array([0.0, 0.0, 0.0])  # mm
            rotation = np.array([rx_deg, ry_deg, 0.0])  # degrees

            # Compute inverse kinematics
            servo_angles = self.ik.inverse_kinematics(translation, rotation)

            if servo_angles is not None:
                # Format command: angle0,angle1,angle2,angle3,angle4,angle5
                cmd = ','.join([f"{angle:.2f}" for angle in servo_angles])
                cmd += '\n'

                # Send command
                self.serial.write(cmd.encode('utf-8'))
                self.servo_command_count += 1

        except Exception as e:
            print(f"Servo command error: {e}")

    def calibrate_gyro(self, duration=3.0):
        """
        Calibrate gyro biases.

        Platform must be stationary during calibration.

        Args:
            duration: Calibration duration in seconds
        """
        print(f"\nCalibrating gyro biases for {duration} seconds...")
        print("Keep platform STATIONARY!")

        self.calibration_mode = True
        self.calibration_samples = []
        self.calibration_duration = duration

        # Wait for samples
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.1)
            sample_count = len(self.calibration_samples)
            print(f"  Collecting samples: {sample_count}", end='\r')

        print(f"\nCollected {len(self.calibration_samples)} samples")

        # Compute biases
        if len(self.calibration_samples) > 100:
            bias_gx, bias_gy, bias_gz = self.kalman.calibrate_biases(self.calibration_samples, duration)
            print(f"Gyro biases: gx={bias_gx:.1f}, gy={bias_gy:.1f}, gz={bias_gz:.1f}")
            print("Calibration complete\n")
        else:
            print("ERROR: Not enough samples for calibration\n")

        self.calibration_mode = False
        self.calibration_samples = []

    def print_status(self):
        """Print current status"""
        elapsed = time.time() - self.start_time

        gyro_hz = self.gyro_count / elapsed if elapsed > 0 else 0
        accel_hz = self.accel_count / elapsed if elapsed > 0 else 0
        servo_hz = self.servo_command_count / elapsed if elapsed > 0 else 0

        total_updates = self.update_count + self.rejected_update_count
        update_rate = self.update_count / total_updates if total_updates > 0 else 0

        print(f"\r[{elapsed:6.1f}s] "
              f"Orientation: rx={self.current_rx:+6.2f}° ry={self.current_ry:+6.2f}° | "
              f"Gyro: {gyro_hz:6.1f}Hz | "
              f"Accel: {accel_hz:6.1f}Hz | "
              f"Update: {update_rate*100:4.1f}% | "
              f"Servo: {servo_hz:5.1f}Hz | "
              f"AccelMag: {self.last_accel_mag:5.2f}m/s²",
              end='')

    def run(self, calibrate=False):
        """
        Run IMU mirror demo.

        Args:
            calibrate: If True, run calibration before starting
        """
        if not self.connected:
            print("ERROR: Not connected")
            return

        # Start threads
        self.running = True

        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        # Wait for initial data
        print("Waiting for IMU data...")
        time.sleep(1.0)

        # Calibration
        if calibrate:
            self.calibrate_gyro(duration=3.0)

        # Reset statistics
        self.start_time = time.time()
        self.gyro_count = 0
        self.accel_count = 0
        self.update_count = 0
        self.rejected_update_count = 0
        self.servo_command_count = 0

        print("IMU Mirror Demo running...")
        print("Platform will mimic IMU orientation")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping...")
            self.running = False


def list_ports():
    """List available serial ports"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found")
        return

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports, 1):
        print(f"  {i}. {port.device} - {port.description}")


def main():
    parser = argparse.ArgumentParser(
        description='IMU Mirror Demo - Platform mimics IMU orientation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_mirror_demo.py --list
  python imu_mirror_demo.py --port COM3
  python imu_mirror_demo.py --port COM3 --calibrate

Note: Upload IMU_control.ino to Teensy first.
      Platform will follow IMU tilt angles with zero translation.
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List available serial ports and exit')
    parser.add_argument('--port', type=str,
                        help='Serial port (e.g., COM3, /dev/ttyACM0)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibrate gyro biases on startup (keep platform still for 3 seconds)')

    args = parser.parse_args()

    # List ports and exit
    if args.list:
        list_ports()
        return

    # Check if port specified
    if not args.port:
        print("ERROR: --port is required")
        print("\nUse --list to see available ports")
        list_ports()
        return

    # Create controller and connect
    controller = IMUMirrorController(args.port)

    try:
        if controller.connect():
            controller.run(calibrate=args.calibrate)
    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()
