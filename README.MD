# Stewart Platform Ball Balancer

6-DOF Stewart platform for ball balancing with camera feedback. PyQt6 GUI, 100Hz hardware control.

```bash
pip install -r requirements.txt

# Simulation
python PID_sim.py
python LQR_sim.py

# Hardware
python PID_real.py
python LQR_real.py
```

## Progress

- [x] System identification (servo τ=0.05s, delay=0.0s)
- [x] Platform kinematics (IK/FK)
- [x] Ball physics simulation (RK4)
- [x] PID controller with Kalman filter
- [x] LQR controller with Kalman filter
- [x] Hardware integration (Pixy2 camera + serial)
- [x] PyQt6 + PyQtGraph GUI (<10ms jitter)
- [ ] Contact dynamics
- [ ] Parallel environments for RL
- [ ] RL controller
