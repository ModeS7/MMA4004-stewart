#!/usr/bin/env python3
"""
Arduino USB Stress Test - Find the Breaking Point

This test progressively increases USB traffic until Windows can't keep up.
Goal: Determine if spikes are caused by traffic volume or Teensy-specific behavior.

Test progression:
1. 100Hz bidirectional (current test) ✓ No spikes
2. 200Hz bidirectional
3. 500Hz bidirectional
4. 1000Hz bidirectional
5. Maximum speed (no sleep)

If Arduino NEVER spikes, it confirms Teensy's 480 Mbit/s implementation is the issue.
"""

import time
import threading
import serial
import serial.tools.list_ports
import numpy as np
import sys
import ctypes
from queue import Queue, Empty


# ============================================================================
# SIMPLIFIED MANAGERS (from previous test)
# ============================================================================

class WindowsTimerManager:
    def __init__(self):
        self.timer_set = False
        self.is_windows = sys.platform.startswith('win')

    def set_high_resolution(self):
        if not self.is_windows:
            return False, "Not Windows"
        try:
            result = ctypes.windll.winmm.timeBeginPeriod(1)
            if result == 0:
                self.timer_set = True
                return True, "Timer set to 1ms"
            return False, f"Failed: {result}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def restore_default(self):
        if self.timer_set:
            try:
                ctypes.windll.winmm.timeEndPeriod(1)
                self.timer_set = False
            except:
                pass


# ============================================================================
# STRESS TEST
# ============================================================================

class StressTest:
    """Progressive stress test to find breaking point."""

    def __init__(self, port):
        self.port = port
        self.serial = None
        self.running = False
        self.thread = None

        self.timer_manager = WindowsTimerManager()

        # Stats
        self.loop_count = 0
        self.spike_count = 0
        self.send_count = 0
        self.recv_count = 0

    def connect(self):
        try:
            self.serial = serial.Serial(self.port, 200000, timeout=0.01)
            time.sleep(2)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.serial and self.serial.is_open:
            self.serial.close()

    def test_frequency(self, target_hz, duration_seconds=30):
        """Test at a specific frequency."""
        print(f"\n{'=' * 70}")
        print(f"Testing at {target_hz} Hz for {duration_seconds} seconds")
        print(f"{'=' * 70}")

        self.loop_count = 0
        self.spike_count = 0
        self.send_count = 0
        self.recv_count = 0

        interval = 1.0 / target_hz if target_hz > 0 else 0.0

        self.running = True
        start_time = time.time()

        while self.running and (time.time() - start_time) < duration_seconds:
            loop_start = time.perf_counter()

            # Send data
            try:
                data = f"{np.random.randn():.4f},{np.random.randn():.4f}\n"
                self.serial.write(data.encode('utf-8'))
                self.send_count += 1
            except:
                pass

            # Receive data
            try:
                if self.serial.in_waiting > 0:
                    self.serial.read(self.serial.in_waiting)
                    self.recv_count += 1
            except:
                pass

            # Check timing
            elapsed = time.perf_counter() - loop_start
            if elapsed > 0.050:  # 50ms spike
                self.spike_count += 1
                print(f"⚠️  SPIKE: {elapsed * 1000:.1f}ms at {target_hz}Hz")

            # Sleep if needed
            if interval > 0:
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.loop_count += 1

        self.running = False

        # Results
        actual_duration = time.time() - start_time
        actual_hz = self.loop_count / actual_duration
        spike_rate = self.spike_count / actual_duration

        print(f"\nResults:")
        print(f"  Target: {target_hz} Hz | Actual: {actual_hz:.1f} Hz")
        print(f"  Total loops: {self.loop_count}")
        print(f"  Spikes: {self.spike_count} ({spike_rate:.2f}/sec)")
        print(f"  Sent: {self.send_count} | Received: {self.recv_count}")

        return spike_rate > 0.5  # True if significant spikes

    def run_progressive_test(self):
        """Run tests at increasing frequencies."""
        print("\n" + "=" * 70)
        print("PROGRESSIVE USB STRESS TEST")
        print("=" * 70)
        print("Goal: Find the frequency where Windows USB driver breaks")
        print("\nStarting LatencyMon now is recommended!")
        input("\nPress Enter to start...")

        # Set timer
        success, msg = self.timer_manager.set_high_resolution()
        print(f"Windows Timer: {msg}")

        test_frequencies = [
            (100, 30),  # 100 Hz for 30s (baseline)
            (200, 30),  # 200 Hz for 30s
            (500, 30),  # 500 Hz for 30s
            (1000, 20),  # 1000 Hz for 20s
            (2000, 20),  # 2000 Hz for 20s (if no spikes yet)
            (0, 20),  # Maximum speed (no sleep) for 20s
        ]

        for freq, duration in test_frequencies:
            if freq == 0:
                print("\n⚠️  WARNING: Maximum speed test (no rate limiting)")
                input("Press Enter to continue...")

            has_spikes = self.test_frequency(freq, duration)

            if has_spikes:
                print(f"\n🎯 BREAKING POINT FOUND at {freq} Hz!")
                print("This tells us the maximum USB traffic Windows can handle")
                break
            else:
                print(f"✅ No spikes at {freq} Hz - continuing...")
                time.sleep(2)  # Brief pause between tests
        else:
            print("\n✅ Arduino NEVER spiked at any frequency!")
            print("This confirms:")
            print("  - Arduino CH340 driver is very efficient")
            print("  - Teensy 4.1's 480 Mbit/s native USB is the issue")
            print("  - Problem is NOT raw traffic volume")

        self.timer_manager.restore_default()


# ============================================================================
# MAIN
# ============================================================================

def find_arduino():
    ports = list(serial.tools.list_ports.comports())

    print("\nAvailable serial ports:")
    for i, port in enumerate(ports):
        print(f"  {i + 1}. {port.device} - {port.description}")

    if not ports:
        return None

    choice = input(f"\nSelect port (1-{len(ports)}) or press Enter for last: ").strip()

    if choice == "":
        return ports[-1].device
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(ports):
                return ports[idx].device
        except:
            pass

    return None


def main():
    print("\n" + "=" * 70)
    print("ARDUINO USB STRESS TEST")
    print("=" * 70)
    print("\nThis test finds the breaking point of Windows USB drivers.")
    print("We'll progressively increase USB traffic until spikes occur.")
    print("\nIf Arduino NEVER spikes (even at 2000Hz), it confirms")
    print("the issue is Teensy 4.1's 480 Mbit/s implementation.")

    port = find_arduino()
    if not port:
        print("No port selected. Exiting.")
        return

    test = StressTest(port)

    if not test.connect():
        print("Failed to connect. Exiting.")
        return

    try:
        test.run_progressive_test()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        test.disconnect()


if __name__ == "__main__":
    main()