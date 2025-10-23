# Stewart Platform Ball Balancer

6-DOF Stewart platform for ball balancing with camera feedback. PyQt6 GUI, 100Hz hardware control.

```bash
pip install -r requirements.txt

# Simulation
python Sim/PID_ball_sim.py
python Sim/LQR_ball_sim.py

# Hardware
python Sim/PID_ball_real.py
python Sim/LQR_ball_real.py
```

## Progress

- [x] System identification (servo τ=0.1s, delay=0.35s)
- [x] Platform kinematics (IK/FK)
- [x] Ball physics simulation (RK4)
- [x] PID controller with Kalman filter
- [x] LQR controller with Kalman filter
- [x] Hardware integration (Pixy2 camera + serial)
- [x] PyQt6 + PyQtGraph GUI (<10ms jitter)
- [ ] Contact dynamics
- [ ] Parallel environments for RL
- [ ] RL controller
