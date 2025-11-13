# Stewart Platform Ball Balancer

Real-time control system for balancing a ball on a 6-DOF Stewart platform using vision feedback and IMU-based tilt compensation.

## Features

- **Control Modes**: PID, LQR, and Manual control with real-time parameter tuning
- **State Estimation**: Kalman filtering for ball position/velocity and IMU orientation
- **Dual Operation**: Identical algorithms run in high-fidelity simulation or on hardware
- **Advanced Features**: Dynamic Z-optimization, IMU tilt correction, 6-DOF yaw tracking and control
- **Real-Time Performance**: 50 Hz control loop with sub-millisecond IK calculations

## Quick Start

```bash
pip install -r requirements.txt

# Full-featured controller with GUI controls
python full_c.py

# Minimal controller (all features enabled by default)
python min_c.py
```

---

## Architecture

### Main Entry Points

#### `full_c.py` - Full Controller
Complete interface with all GUI controls visible. Supports mode switching (simulation/hardware), controller selection (PID/LQR/Manual), and real-time parameter adjustment.

**Features**:
- Mode selector (simulation/hardware)
- Controller type switching
- Ball position Kalman filter controls
- IMU orientation parameters
- Dynamic Z-optimization toggle
- Performance data recording
- Real-time plotting

**Default Settings**:
- Ball Kalman filter: **Enabled**
- Z-optimization: **Disabled** (GUI toggle)
- IMU tilt correction: **Disabled** (GUI toggle)
- IMU yaw tracking: **Disabled** (4-DOF mode, GUI toggle for 6-DOF)
- PID Kalman derivative: **Disabled** (GUI toggle)

---

#### `min_c.py` - Minimal Controller
Streamlined interface for rapid development. All advanced features enabled by default with minimal GUI clutter.

**Features**:
- Essential controls only (mode, controller, trajectory, manual pose)
- Single scrollable column layout
- No display widgets (ball state, servo angles, platform pose hidden)
- All settings enabled invisibly

**Default Settings**:
- Ball Kalman filter: **Enabled**
- Z-optimization: **Enabled**
- IMU tilt correction: **Enabled**
- IMU yaw tracking: **Disabled** (4-DOF mode)
- PID Kalman derivative: **Enabled**

---

### Core Modules (`core/`)

Foundation algorithms and models used by both simulation and hardware modes.

#### `core/core.py`
Physical system models and kinematics:
- **`FirstOrderServo`**: Servo dynamics with delay, time constant (τ=50ms), velocity limiting
- **`StewartPlatformIK`**: Inverse kinematics (closed-form) and forward kinematics (Newton-Raphson)
- **`SimpleBallPhysics2D`**: RK4 integration with rolling friction, air resistance, platform dynamics
- **`Pixy2Camera`**: Camera noise model with pixel quantization (1.4mm V1, 2.0mm V2), 19.3 Hz sampling
- **`TrajectoryPattern`**: Circle, figure-8, star, and static target patterns

#### `core/control_core.py`
Controllers and state estimation:
- **`PIDController`**: 2D PID with integral limiting, optional derivative filtering
- **`LQRController`**: State-feedback using continuous-time algebraic Riccati equation (CARE)
- **`KalmanFilter`**: Ball position/velocity estimation with platform tilt dynamics prediction
- **`OrientationKalmanFilter`**: IMU-based 6-DOF orientation (Extended Kalman Filter)
- **`IMUControllerMixin`**: Hardware IMU integration with calibration sequences
- **`clip_tilt_vector()`**: Direction-preserving magnitude limiting

#### `core/utils.py`
Configuration constants and parameters:
- **Platform geometry**: Dimensions, servo limits, tilt constraints, yaw limits
- **Controller defaults**: PID/LQR gains for simulation and hardware
- **Physics config**: Ball properties, friction, drag coefficients
- **Noise parameters**: Kalman filter process/measurement noise, camera noise
- **IMU calibration**: Gyro bias, sensor noise, magnetometer offsets
- **Control loop timing**: Frequency (50 Hz default), intervals, timeouts

---

### GUI System (`gui/`)

Modular PyQt6 interface with declarative layout system.

#### `gui/gui_builder.py`
Layout construction from configuration dictionaries:
- **`GUIBuilder`**: Creates GUI from layout config and module registry
- **`ScrollableColumn`**: QScrollArea-based column container for overflow handling
- **`create_standard_layout()`**: Template for 2-column layouts with plot panel

**Layout Config Format**:
```python
{
    'columns': [
        {
            'width': 400,
            'scrollable': True,
            'modules': [
                {'type': 'simulation_control'},
                {'type': 'controller', 'args': {...}},
                ...
            ]
        }
    ],
    'plot': {'enabled': True, 'title': 'Ball Position'}
}
```

#### `gui/gui_modules.py`
20+ reusable GUI components:

**Control Modules**:
- `SimulationControlModule`: Start/stop, reset controls
- `ControllerModule`: PID/LQR parameter sliders with scalar multipliers
- `TrajectoryPatternModule`: Pattern selection and parameters
- `BallControlModule`: Ball reset and push controls
- `ManualPoseControlModule`: 6-DOF manual positioning

**Display Modules**:
- `BallStateModule`: Position, velocity, error display
- `ServoAnglesModule`: Commanded and actual servo angles
- `PlatformPoseModule`: X, Y, Z, RX, RY, RZ state
- `ControllerOutputModule`: Controller tilt commands
- `DebugLogModule`: Timestamped message log

**Configuration Modules**:
- `ModeSelectionModule`: Simulation/hardware mode switcher
- `ControllerSelectionModule`: PID/LQR/Manual selector
- `ConfigurationModule`: Top surface offset toggle
- `PlotControlModule`: Plot rate and enable/disable
- `KalmanFilterModule`: Process/measurement noise, reset
- `IMUKalmanParametersModule`: IMU filter parameters, 4-DOF/6-DOF toggle, yaw calibration
- `IKZOptimizationModule`: Dynamic Z-optimization control

**Hardware Modules**:
- `SerialConnectionModule`: COM port connection
- `PerformanceStatsModule`: Loop timing statistics
- `ControlFrequencyModule`: Control loop frequency (50-500 Hz)

**Data Collection**:
- `PerformanceDataCollectionModule`: CSV recording for analysis

---

### Setup Classes (`setup/`)

Base classes for simulation and hardware modes with shared abstractions.

#### `setup/base_simulator.py`
**`BaseStewartSimulator(QMainWindow)`**: Foundation for all simulation modes.

**Core Responsibilities**:
- PyQt6 window management and modular GUI building
- Physics simulation loop with servo dynamics
- Ball physics integration (RK4)
- Camera noise simulation
- Trajectory tracking
- IK/FK calculation with caching
- Controller abstraction (subclasses implement PID/LQR/Manual)

**Key Methods**:
- `_build_modular_gui()`: Constructs layout from config
- `_simulation_loop()`: Physics update at 500 Hz (2ms intervals)
- `_control_loop()`: Controller update at configurable rate
- `_update_controller()`: Abstract method for controller logic
- `update_gui_modules()`: Refreshes all GUI widgets with current state

#### `setup/base_hardware.py`
**`HardwareControllerBase(BaseStewartSimulator)`**: Extends simulator for real hardware.

**Hardware Integration**:
- Serial communication with servos (115200 baud)
- Pixy2 camera data acquisition via serial
- IMU data streaming (accelerometer, gyroscope, magnetometer)
- Thread-based control loop with high-resolution timing
- Windows multimedia timer (1ms resolution)

**Key Components**:
- **`SerialController`**: Manages servo communication and IMU data parsing
- **`_control_thread_func()`**: Real-time control loop running in separate thread
- **`_gui_update_loop()`**: QTimer-based GUI updates (5 Hz)
- **IK caching**: >95% cache hit rate for performance

**Differences from Simulation**:
- No physics simulation (uses real ball tracking)
- Serial I/O instead of synthetic data
- Thread-based control (vs QTimer)
- Actual servo feedback (FK from commanded angles)

---

## System Flow

### Simulation Mode
```
Camera Model → Ball Physics → Controller → IK → Servo Model → FK → Ball Physics
     ↓              ↓             ↓                                      ↑
  Kalman       Trajectory    PID/LQR                              Update Loop
```

### Hardware Mode
```
Pixy2 Camera → Kalman Filter → Controller → IK → Serial (Servos) → Platform
     ↓              ↓             ↓         ↓           ↓              ↓
  19.3 Hz       50 Hz State   PID/LQR  Optimize Z   FK (verify)    Move Ball
                             + IMU Tilt
                             + Yaw Ctrl
```

---

## Control Algorithms

### PID Controller
2D independent PID loops for X and Y axes:
- **Proportional**: Error-based correction
- **Integral**: Steady-state error elimination (with anti-windup)
- **Derivative**: Damping (optional Kalman velocity for smoothness)

**Tuning**: Separate gains for simulation (aggressive) and hardware (conservative)

### LQR Controller
State-space optimal control using Linear Quadratic Regulator:
- **State**: [x, y, vx, vy] - position and velocity
- **Control**: [rx, ry] - platform tilt angles (yaw controlled separately via IMU)
- **Cost**: Weighted position error, velocity, and control effort
- **Gain**: Computed via algebraic Riccati equation (scipy)

**Advantages**: Optimal for linear systems, explicit velocity feedback

---

## Key Features Detail

### Dynamic Z-Optimization
Automatically adjusts platform Z height to balance servo angles around neutral (0°):
- Maximizes available servo range
- Prevents saturation
- Uses iterative optimization (25 iterations max, 0.1° tolerance)
- Falls back to standard IK if optimization fails

### IMU Tilt Correction
Compensates for base platform tilt using orientation Kalman filter:
- Measures platform tilt in gravity frame
- Adds correction to controller output: `tilt_final = tilt_controller + (-tilt_imu * gain)`
- Ensures ball sees desired effective tilt regardless of base orientation
- Clipped to ±15° for safety
- **Yaw Control**: When 6-DOF mode enabled, platform rotates to match IMU yaw (±45° limit)

### Kalman Filtering

#### Ball Position/Velocity Filter
Predicts ball dynamics using platform tilt from previous timestep:
- **Process model**: Ball physics with gravity, friction, platform acceleration
- **Measurement**: Noisy camera position
- **Output**: Smooth position and velocity estimates

#### IMU Orientation Filter
Extended Kalman Filter for orientation estimation:
- **4-DOF Mode (default)**: Roll and pitch only using accelerometer + gyroscope
  - State: [roll, pitch, gyro_bias_x, gyro_bias_y]
- **6-DOF Mode**: Full orientation with magnetometer-based yaw tracking
  - State: [roll, pitch, yaw, gyro_bias_x, gyro_bias_y, gyro_bias_z]
  - Auto-calibrates yaw reference on enable
  - Manual calibration: "Set Current as 0°" button or offset slider (±180°)
  - Decoupled magnetometer update prevents corruption of roll/pitch estimates
- **Features**: Gyroscope bias estimation, tilt-compensated heading

---

## Default Parameters

### Control
- **Frequency**: 50 Hz (20ms intervals)
- **PID Gains (Sim)**: Kp=3.0, Ki=0.0, Kd=3.0
- **PID Gains (HW)**: Kp=1.0, Ki=0.0, Kd=4.0
- **LQR Weights**: Q_pos=10.0, Q_vel=1.0, R=1.0
- **Max Tilt Angle**: 15° (roll/pitch)
- **Max Yaw Angle**: 45° (rotation)

### Kalman Filters
- **Ball Process Noise**: 1.0
- **Ball Measurement Noise**: 1.0
- **IMU Accel Noise**: 1.0 m/s²
- **IMU Gyro Noise**: 0.0224 rad/s

### Camera (V1)
- **Pixel Size**: 1.4 mm
- **Subpixel Noise**: 0.4 mm (before quantization)
- **Sample Rate**: 19.3 Hz
- **Detection Rate**: 99.9%

### Servo Dynamics
- **Time Constant (τ)**: 50 ms
- **Delay**: 40 ms (sim), 0 ms (hardware)
- **Max Velocity**: 545°/s

---

## Hardware Requirements

- **Controller**: Teensy 4.1 (or similar microcontroller)
- **Vision**: Pixy2 (CMUcam5) camera
- **IMU**: LSM303 (accel/mag) + L3GD20 (gyro)
- **Servos**: 6x high-speed servos (e.g., E-Power D5545GS)
- **Servo Controller**: Pololu Maestro 12-channel
- **Platform**: V1 (200x200mm square) or V2 (larger circular)

## Software Dependencies

```
PyQt6>=6.0.0
pyqtgraph>=0.13.0
numpy>=1.20.0
scipy>=1.7.0
pyserial>=3.5
```