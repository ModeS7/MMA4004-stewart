#!/usr/bin/env python3
"""
IMU Data Logger for Kalman Filter Tuning

Connects to Arduino running IMU_test.ino and logs IMU data to CSV.
Data includes accelerometer, gyroscope, and magnetometer (LSM303 + L3GD20) readings for Kalman filter tuning.

IMU_test.ino output format:
- Accelerometer: "A:ax,ay,az"
- Gyroscope: "G:gx,gy,gz"
- Magnetometer: "M:mx,my,mz"
- Rate reports: "RATE - Accel: xx.xx Hz | Gyro: xx.xx Hz | Mag: xx.xx Hz"
"""

import serial
import serial.tools.list_ports
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path


class IMULogger:
    def __init__(self, port, baudrate=2000000):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.csv_writer = None
        self.csv_file = None
        self.last_accel = [0, 0, 0]
        self.last_gyro = [0, 0, 0]
        self.last_mag = [0, 0, 0]

    def connect(self):
        """Connect to Arduino running IMU_test.ino"""
        try:
            print(f"Connecting to {self.port} at {self.baudrate} baud...")
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
            time.sleep(2.5)  # Wait for Arduino to reset

            # Read startup messages
            startup_messages = []
            start_wait = time.time()
            while time.time() - start_wait < 2.0:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        startup_messages.append(line)
                        print(f"  Arduino: {line}")

            # Check if sampling started
            if any("Sampling started" in msg for msg in startup_messages):
                print("  IMU sampling active")
            else:
                print("  WARNING: Expected 'Sampling started' message not detected")

            self.is_connected = True
            print("Connection established")
            print("Accelerometer: LSM303 (~1344 Hz)")
            print("Gyroscope: L3GD20 (~800 Hz)")
            print("Magnetometer: LSM303 (~220 Hz)\n")
            return True

        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from Arduino"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False

    def start_logging(self, output_file, duration=None):
        """Start logging IMU data to CSV file"""
        if not self.is_connected:
            print("ERROR: Not connected to Arduino")
            return

        # Create output directory if needed
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Open CSV file
            self.csv_file = open(output_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

            # Write header with metadata
            self.csv_writer.writerow(['# IMU Data Log - LSM303 + L3GD20'])
            self.csv_writer.writerow([f'# Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
            self.csv_writer.writerow([f'# Port: {self.port}'])
            self.csv_writer.writerow([f'# Baud: {self.baudrate}'])
            self.csv_writer.writerow([f'# Duration: {duration}s' if duration else '# Duration: Manual stop'])
            self.csv_writer.writerow(['# Accelerometer: LSM303 (12-bit, ~1344 Hz)'])
            self.csv_writer.writerow(['# Gyroscope: L3GD20 (16-bit, ~800 Hz)'])
            self.csv_writer.writerow(['# Magnetometer: LSM303 (16-bit, ~220 Hz)'])
            self.csv_writer.writerow([])

            # Write column headers
            self.csv_writer.writerow(['timestamp_pc', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz'])

            print(f"Logging to: {output_file}")
            if duration:
                print(f"Duration: {duration} seconds")
            else:
                print("Press Ctrl+C to stop logging\n")

            # Clear any buffered data
            self.serial_conn.reset_input_buffer()

            # Start logging
            start_time = time.time()
            sample_count = 0
            accel_count = 0
            gyro_count = 0
            mag_count = 0
            last_status_time = start_time

            while True:
                # Check duration limit
                if duration and (time.time() - start_time) >= duration:
                    break

                # Read serial data
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()

                    if line.startswith("A:"):
                        # Parse accelerometer: A:ax,ay,az
                        parts = line[2:].split(',')
                        if len(parts) == 3:
                            try:
                                timestamp_pc = time.time()
                                self.last_accel = [int(parts[0]), int(parts[1]), int(parts[2])]

                                # Write combined row
                                self.csv_writer.writerow([
                                    timestamp_pc,
                                    self.last_accel[0], self.last_accel[1], self.last_accel[2],
                                    self.last_gyro[0], self.last_gyro[1], self.last_gyro[2],
                                    self.last_mag[0], self.last_mag[1], self.last_mag[2]
                                ])

                                sample_count += 1
                                accel_count += 1

                            except ValueError:
                                pass

                    elif line.startswith("G:"):
                        # Parse gyroscope: G:gx,gy,gz
                        parts = line[2:].split(',')
                        if len(parts) == 3:
                            try:
                                self.last_gyro = [int(parts[0]), int(parts[1]), int(parts[2])]
                                gyro_count += 1
                            except ValueError:
                                pass

                    elif line.startswith("M:"):
                        # Parse magnetometer: M:mx,my,mz
                        parts = line[2:].split(',')
                        if len(parts) == 3:
                            try:
                                self.last_mag = [int(parts[0]), int(parts[1]), int(parts[2])]
                                mag_count += 1
                            except ValueError:
                                pass

                    elif line.startswith("RATE"):
                        # Rate report from Arduino
                        print(f"  {line}")

                # Print status every second
                if time.time() - last_status_time >= 1.0:
                    elapsed = time.time() - start_time
                    accel_rate = accel_count / elapsed if elapsed > 0 else 0
                    gyro_rate = gyro_count / elapsed if elapsed > 0 else 0
                    mag_rate = mag_count / elapsed if elapsed > 0 else 0
                    print(f"  Samples: {sample_count:6d} | "
                          f"Elapsed: {elapsed:6.1f}s | "
                          f"Accel: {accel_rate:6.1f} Hz | "
                          f"Gyro: {gyro_rate:6.1f} Hz | "
                          f"Mag: {mag_rate:6.1f} Hz", end='\r')
                    last_status_time = time.time()

            # Final summary
            elapsed = time.time() - start_time
            accel_rate = accel_count / elapsed if elapsed > 0 else 0
            gyro_rate = gyro_count / elapsed if elapsed > 0 else 0
            mag_rate = mag_count / elapsed if elapsed > 0 else 0
            print(f"\n\nLogging complete:")
            print(f"  Total samples: {sample_count}")
            print(f"  Accelerometer: {accel_count} samples ({accel_rate:.2f} Hz)")
            print(f"  Gyroscope: {gyro_count} samples ({gyro_rate:.2f} Hz)")
            print(f"  Magnetometer: {mag_count} samples ({mag_rate:.2f} Hz)")
            print(f"  Duration: {elapsed:.2f} seconds")
            print(f"  Saved to: {output_file}")

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            accel_rate = accel_count / elapsed if elapsed > 0 else 0
            gyro_rate = gyro_count / elapsed if elapsed > 0 else 0
            mag_rate = mag_count / elapsed if elapsed > 0 else 0
            print(f"\n\nLogging stopped by user:")
            print(f"  Total samples: {sample_count}")
            print(f"  Accelerometer: {accel_count} samples ({accel_rate:.2f} Hz)")
            print(f"  Gyroscope: {gyro_count} samples ({gyro_rate:.2f} Hz)")
            print(f"  Magnetometer: {mag_count} samples ({mag_rate:.2f} Hz)")
            print(f"  Duration: {elapsed:.2f} seconds")
            print(f"  Saved to: {output_file}")

        except Exception as e:
            print(f"\nERROR during logging: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if self.csv_file:
                self.csv_file.close()

    def cleanup(self):
        """Clean up resources"""
        if self.csv_file:
            self.csv_file.close()
        self.disconnect()


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
        description='Log IMU data from Arduino (IMU_test.ino) to CSV for Kalman filter tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_logger.py --list
  python imu_logger.py --port COM3 --duration 60
  python imu_logger.py --port /dev/ttyACM0 --duration 30
  python imu_logger.py --port COM3 --output my_data.csv

Note: IMU_test.ino must be uploaded to Arduino first.
      Uses 2000000 baud rate (2 Mbps).
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List available serial ports and exit')
    parser.add_argument('--port', type=str,
                        help='Serial port (e.g., COM3, /dev/ttyACM0)')
    parser.add_argument('--duration', type=float,
                        help='Logging duration in seconds (default: manual stop with Ctrl+C)')
    parser.add_argument('--output', type=str,
                        help='Output CSV file (default: auto-generated with timestamp)')

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

    # Generate output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"imu_data_{timestamp}.csv"

    # Create logger and connect (fixed 2 Mbaud for IMU_test.ino)
    logger = IMULogger(args.port, baudrate=2000000)

    try:
        if logger.connect():
            logger.start_logging(args.output, args.duration)
    finally:
        logger.cleanup()


if __name__ == "__main__":
    main()
