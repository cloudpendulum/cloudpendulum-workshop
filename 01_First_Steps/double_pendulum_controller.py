"""Hidden helper for the double pendulum example in tutorial.ipynb.

Gravity-compensated PID trajectory tracker (the double pendulum version of the
GravityPIDController from tutorial 03). Angles: q1 absolute, q2 relative to
link 1, 0 = hanging down, pi = pointing up.
"""
import numpy as np


class GravityPIDTrajectoryController:
    def __init__(self, traj, dt, torque_limit=(0.14, 0.14),
                 Kp=(0.1, 0.1), Ki=(0.02, 0.02), Kd=(0.004, 0.004),
                 mass=(0.131, 0.064), length=(0.05, 0.05), com=(0.046, 0.048),
                 gravity=9.81):
        self.traj = np.asarray(traj, dtype=float)   # shape (n, 2)
        self.dt = dt
        self.tl = np.asarray(torque_limit, dtype=float)
        self.Kp, self.Ki, self.Kd = (np.asarray(k, dtype=float) for k in (Kp, Ki, Kd))
        self.m, self.l, self.r, self.g = mass, length, com, gravity
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = None

    def gravity_torque(self, q):
        m1, m2 = self.m
        l1, _ = self.l
        r1, r2 = self.r
        g = self.g
        g2 = m2 * r2 * g * np.sin(q[0] + q[1])
        g1 = (m1 * r1 + m2 * l1) * g * np.sin(q[0]) + g2
        return np.array([g1, g2])

    def get_control_output(self, q, i):
        """Torque [tau1, tau2] for measured positions q at control step i."""
        q = np.asarray(q, dtype=float)
        q_des = self.traj[min(i, len(self.traj) - 1)]
        error = q_des - q
        # integral with simple anti-windup
        self.integral = np.clip(self.integral + error * self.dt, -0.5, 0.5)
        deriv = np.zeros(2) if self.prev_error is None else (error - self.prev_error) / self.dt
        self.prev_error = error
        tau = (self.gravity_torque(q) + self.Kp * error
               + self.Ki * self.integral + self.Kd * deriv)
        return np.clip(tau, -self.tl, self.tl)
