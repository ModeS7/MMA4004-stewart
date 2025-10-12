1.
done - Run step response tests on all 6 servos to extract time constant τ
done - Measure actual platform geometry (link lengths, joint offsets)
done - Record test trajectories: commanded angles + IMU data
done - Implement first-order servo dynamics K = 1, τ = 0.1s, delay = 0.35s

2.
done - Make a working sim with FK
- implement ball dynamics
- implement contact dynamics

3.
- Run recorded trajectories in simulation (open-loop)
- Compare simulated vs real IMU traces (plot overlays)
- Tune parameters in order:
    - Contact parameters (friction, damping)
    - Ball properties (mass, restitution)
    - Servo time constant (if needed)
4.
- Implement LQR/PID controller
- Test on real hardware to establish baseline

5.
- Set up parallel environments
- Implement RL
- Test on real hardware to establish baseline
