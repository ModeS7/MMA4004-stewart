#!/usr/bin/env python3
"""
Hardware Stewart Platform Controller Configuration
Real-time control with Pixy2 camera and Maestro servos

Key differences from simulation:
- Real camera data (no physics simulation)
- Serial communication with hardware
- 100Hz dedicated control thread
- Camera Y-axis inversion (ry = -ry)
- Performance optimization with IK caching
- Derivative filtering for camera noise reduction
- High-speed serial (200000 baud)
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import threading
import serial
import serial.tools.list_ports
from queue import Queue, Empty

from setup.base_simulator import ControllerConfig
from core.control_core import PIDController
from core.utils import ControlLoopConfig, GUIConfig


class IKCache:
    """Cache IK results with coarse resolution for higher hit rate."""

    def __init__(self, max_size=5000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get_key(self, translation, rotation):
        t_key = tuple(np.round(translation, 0))
        r_key = tuple(np.round(rotation, 0))
        return (t_key, r_key)

    def get(self, translation, rotation):
        key = self.get_key(translation, rotation)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, translation, rotation, angles):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        key = self.get_key(translation, rotation)
        self.cache[key] = angles.copy()

    def get_hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class SerialController:
    """High-performance serial communication with hardware."""

    def __init__(self, port, baudrate=200000):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.connected = False
        self.read_thread = None
        self.write_thread = None
        self.running = False

        self.ball_data_queue = Queue(maxsize=10)
        self.command_queue = Queue(maxsize=20)
        self.last_command_time = 0

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate,
                                        timeout=0.1, write_timeout=0.5)
            time.sleep(2)
            self.connected = True

            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()

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
        self.running = False

        if self.read_thread:
            self.read_thread.join(timeout=1.0)
        if self.write_thread:
            self.write_thread.join(timeout=1.0)

        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False

    def _read_loop(self):
        buffer = ""

        while self.running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting > 0:
                    chunk = self.serial.read(self.serial.in_waiting).decode(
                        'utf-8', errors='ignore'
                    )
                    buffer += chunk

                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()

                        if line.startswith("BALL:"):
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

                                    if self.ball_data_queue.full():
                                        try:
                                            self.ball_data_queue.get_nowait()
                                        except Empty:
                                            pass
                                    self.ball_data_queue.put(ball_data)
                            except (ValueError, IndexError):
                                pass

                time.sleep(0.0005)

            except Exception as e:
                if self.running:
                    print(f"Serial read error: {e}")
                time.sleep(0.1)

    def _write_loop(self):
        error_count = 0
        max_errors = 5

        while self.running and self.serial and self.serial.is_open:
            try:
                try:
                    command = self.command_queue.get(timeout=0.01)
                    self.serial.write(command.encode('utf-8'))
                    self.last_command_time = time.time()
                    error_count = 0
                    time.sleep(0.003)

                except Empty:
                    pass

            except serial.SerialTimeoutException:
                error_count += 1
                if error_count > max_errors:
                    while not self.command_queue.empty():
                        try:
                            self.command_queue.get_nowait()
                        except Empty:
                            break
                    error_count = 0
                time.sleep(0.1)

            except Exception as e:
                if self.running:
                    error_count += 1
                time.sleep(0.1)

    def send_servo_angles(self, angles):
        """Queue servo angles with rate limiting."""
        if not self.connected:
            return False

        try:
            current_time = time.time()
            time_since_last = current_time - self.last_command_time

            queue_size = self.command_queue.qsize()
            if queue_size > 10:
                min_interval = 0.05
            elif queue_size > 5:
                min_interval = 0.02
            else:
                min_interval = ControlLoopConfig.INTERVAL_S

            if time_since_last < min_interval:
                return True

            command = ','.join([f'{angle:.2f}' for angle in angles]) + '\n'

            if self.command_queue.qsize() >= 15:
                return True

            self.command_queue.put_nowait(command)
            return True
        except Exception as e:
            print(f"Error queueing command: {e}")
            return False

    def send_command(self, cmd):
        if not self.connected:
            return False
        try:
            if self.command_queue.qsize() >= 15:
                return False
            self.command_queue.put_nowait(cmd + '\n')
            return True
        except:
            return False

    def set_servo_speed(self, speed):
        return self.send_command(f'SPD:{speed}')

    def set_servo_acceleration(self, accel):
        return self.send_command(f'ACC:{accel}')

    def get_latest_ball_data(self):
        try:
            data = None
            while not self.ball_data_queue.empty():
                data = self.ball_data_queue.get_nowait()
            return data
        except Empty:
            return None


class HardwareControllerConfig(ControllerConfig):
    """Hardware PID controller configuration with derivative filtering."""

    def __init__(self):
        self.scalar_values = [0.0000001, 0.000001, 0.00001, 0.0001,
                              0.001, 0.01, 0.1, 1.0, 10.0]
        self.default_gains = {'kp': 3.0, 'ki': 1.0, 'kd': 3.0}
        self.default_scalar_idx = 3

    def get_controller_name(self) -> str:
        return "PID (Hardware)"

    def create_controller(self, **kwargs):
        return PIDController(
            kp=kwargs.get('kp', 0.0003),
            ki=kwargs.get('ki', 0.0001),
            kd=kwargs.get('kd', 0.0003),
            output_limit=kwargs.get('output_limit', 15.0),
            derivative_filter_alpha=kwargs.get('derivative_filter_alpha', 0.1)
        )

    def get_scalar_values(self) -> list:
        return self.scalar_values


def main():
    """Launch hardware controller."""
    print("Hardware controller configuration loaded (200000 baud).")
    print("Use HardwareStewartSimulator to run with real hardware.")


if __name__ == "__main__":
    main()