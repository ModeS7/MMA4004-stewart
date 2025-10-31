# Stewart Platform Ball Balancer

6-DOF Stewart platform for ball balancing with vision feedback. Real-time control at 100-250 Hz with IMU-based tilt compensation.

## Quick Start

```bash
pip install -r requirements.txt

# Full-featured controller (simulation or hardware)
python full_c.py

# Minimal controller (faster iteration)
python min_c.py
```

## Features

- **Controllers**: PID, LQR, Manual control
- **State Estimation**: Kalman filter for ball position/velocity
- **IMU Integration**: Real-time tilt compensation with orientation Kalman filter
- **Hardware**: Teensy 4.1 + Pixy2 camera + Pololu Maestro servo controller
- **GUI**: PyQt6 with 20+ modular components, real-time plotting
- **Simulation**: High-fidelity physics with RK4 integration

## Architecture

```
core/              # Control algorithms, IK/FK, physics simulation
gui/               # Modular PyQt6 interface
setup/             # Simulation & hardware control base classes
imu/               # Real-time orientation tracking with Kalman filtering
misc/arduino/      # Teensy 4.1 firmware
```

## Hardware Stack

- **Teensy 4.1**: Main controller (600 MHz ARM Cortex-M7)
- **Pixy2 (CMUcam5)**: Ball tracking at 50 Hz
- **Pololu Maestro**: 6-channel servo controller
- **LSM303 + L3GD20**: IMU (accelerometer, gyroscope, magnetometer)
- **6x Servos**: Platform actuation

## Key Capabilities

- Unified simulation/hardware control with identical algorithms
- Dynamic Z-axis optimization for mechanical efficiency
- Impact rejection during rapid platform motion
- IK result caching for performance (>95% hit rate)
- High-resolution timing and thread priority management
