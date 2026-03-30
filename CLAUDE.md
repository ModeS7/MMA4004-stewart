# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stewart Platform Ball Balancer — a real-time control system for a 6-DOF parallel manipulator that balances a ball on a plate using vision feedback (Pixy2 camera) and IMU-based tilt compensation. Runs in both high-fidelity simulation and on real hardware.

## Running the Application

```bash
pip install -r requirements.txt

# Full controller — all GUI controls visible, features toggled manually
python full_c.py

# Minimal controller — streamlined GUI, all advanced features enabled by default
python min_c.py
```

There is no test suite, linter, or CI pipeline. Validation is done through simulation runs and hardware testing.

## Architecture

### Layered Module Design

The codebase follows a strict layering: **core** (pure algorithms) → **setup** (base classes) → **entry points** (`full_c.py`, `min_c.py`), with **gui** as a side module consumed by setup.

```
full_c.py / min_c.py          ← Entry points (extend HardwareControllerBase)
  └── setup/base_hardware.py  ← HardwareControllerBase (thread-based control, serial I/O)
        └── setup/base_simulator.py  ← BaseStewartSimulator (QMainWindow, physics loop)
              ├── core/core.py         ← IK/FK, servo dynamics, ball physics, camera model
              ├── core/control_core.py ← PID, LQR, Kalman filters, IMU mixin
              ├── core/utils.py        ← All configuration dataclasses
              └── gui/                 ← Modular PyQt6 widget system
```

### Key Inheritance Chain

`BaseStewartSimulator(QMainWindow)` → `HardwareControllerBase` → `StewartController` (in full_c/min_c)

- `BaseStewartSimulator` owns physics simulation, trajectory generation, GUI building, and defines abstract `_update_controller()`
- `HardwareControllerBase` adds serial communication, threaded control loop, IK caching, Windows timer management
- Entry point classes implement controller logic and define GUI layout configs

### GUI System

Declarative layout system: entry points define a layout config dict → `GUIBuilder` constructs the UI from a registry of 20+ modular widgets in `gui/gui_modules.py`. Widgets communicate via callbacks, not direct coupling.

### Control Flow

- **Simulation**: QTimer drives physics at 500 Hz and control at 50 Hz. Ball physics uses RK4 integration with servo dynamics.
- **Hardware**: Dedicated thread runs control loop at 50 Hz with Windows 1ms timer resolution. Serial I/O at 115200 baud for servos and sensor data.

### Independent Subsystems

- **`RL/`** — Reinforcement learning (SAC/PPO) with custom Gymnasium environments (`StewartEnv`, `StewartEnvVec`). Uses the same ball physics from `core/core.py`. Train with `python RL/train_sac.py` or `python RL/train_ppo.py`.
- **`ball_detection/`** — CNN-based vision pipeline for ball detection with stereo triangulation. Has its own training (`ball_detection/training/train.py`), ONNX export, calibration tools, and benchmarking utilities.
- **`imu/`** — Standalone IMU data logging and rotation utilities.

## Key Design Patterns

- **Configuration-driven**: All system parameters live as dataclasses in `core/utils.py` (platform geometry, controller gains, physics constants, noise models). Modify parameters there, not scattered through code.
- **IK caching**: `IKCache` in `base_hardware.py` uses coarse-resolution keys for >95% hit rate on hardware.
- **Z-optimization**: Iterative adjustment of platform height to balance servo angles around neutral — implemented in `StewartPlatformIK`.
- **IMU tilt correction**: `IMUControllerMixin` in `control_core.py` adds measured tilt as correction to controller output.
- **Numba JIT**: Ball physics batch processing uses `@njit` for performance-critical paths.

## Important Conventions

- Simulation and hardware use separate PID gains (sim is aggressive, hardware is conservative) — both defined in `PIDConfig` and `LQRConfig`.
- The `_update_controller()` method is the main extension point — override it in entry point classes.
- GUI modules are self-contained widgets registered by string key in `GUIBuilder`. Add new modules to `gui/gui_modules.py` and reference them in layout configs.
- Servo angles, platform tilts, and yaw are all in degrees throughout the codebase.
