#!/usr/bin/env python3
"""
Trajectory optimization for the double pendulum using CasADi + IPOPT.

Solves a multiple-shooting NLP to find minimum-effort torques that swing
the pendulum from the hanging position (q=[0,0]) to the upright position
(q=[pi,0]) in 5 s. Decision variables are sampled at dt=0.001 s so the
solution is written directly to trajectory_modelbased.csv without any
post-hoc re-simulation or interpolation.
"""

import sys
import os
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from double_pendulum.model.plant import DoublePendulumPlant


def upsample_trajectory(t, X, U, dt_new):
    """
    Upsample a trajectory to a finer time grid using cubic polynomial interpolation.

    Parameters
    ----------
    t      : (N+1,) original time grid
    X      : (n_x, N+1) state array
    U      : (n_u, N+1) input array (last column may be zero-padded)
    dt_new : desired sampling period of the output

    Returns
    -------
    t_new : (M+1,) upsampled time grid
    X_new : (n_x, M+1) upsampled states
    U_new : (n_u, M+1) upsampled inputs
    """
    t_new = np.arange(t[0], t[-1] + dt_new / 2, dt_new)
    X_new = np.vstack([CubicSpline(t, X[i])(t_new) for i in range(X.shape[0])])
    U_new = np.vstack([CubicSpline(t, U[i])(t_new) for i in range(U.shape[0])])
    return t_new, X_new, U_new

def trajectory_optimization(mass, length, com, inertia, damping, gravity, tau_max, dt, t_final, csvfile=None):
    # Trajectory parameters
    x0 = np.array([0.0,0.0, 0.0, 0.0])   # initial state [q1, q2, dq1, dq2]
    xf = np.array([np.pi, 0.0, 0.0, 0.0])   # final state

    N = int(t_final/ dt)

    ## Dynamics 
    m, l, r = mass, length, com
    I, b, g = inertia, damping, gravity

    x_sym = ca.SX.sym('x', 4)
    u_sym = ca.SX.sym('u', 2)

    q1, q2   = x_sym[0], x_sym[1]
    dq1, dq2 = x_sym[2], x_sym[3]

    # Mass matrix
    m00 = I[0] + I[1] + m[1]*l[0]**2 + 2*m[1]*l[0]*r[1]*ca.cos(q2)
    m01 = I[1] + m[1]*l[0]*r[1]*ca.cos(q2)
    M_ca = ca.vertcat(ca.horzcat(m00, m01),
                    ca.horzcat(m01, I[1]))

    # Coriolis matrix
    C_ca = ca.vertcat(
        ca.horzcat(-2*m[1]*l[0]*r[1]*ca.sin(q2)*dq2,
                    -m[1]*l[0]*r[1]*ca.sin(q2)*dq2),
        ca.horzcat(  m[1]*l[0]*r[1]*ca.sin(q2)*dq1,  0.0)
    )

    # Gravity vector
    G_ca = ca.vertcat(
        -m[0]*g*r[0]*ca.sin(q1) - m[1]*g*(l[0]*ca.sin(q1) + r[1]*ca.sin(q1+q2)),
        -m[1]*g*r[1]*ca.sin(q1+q2)
    )

    # Viscous damping
    F_ca = ca.vertcat(b[0]*dq1, b[1]*dq2)

    vel_sym = ca.vertcat(dq1, dq2)
    qdd     = ca.solve(M_ca, G_ca + u_sym - ca.mtimes(C_ca, vel_sym) - F_ca)
    xdot    = ca.vertcat(dq1, dq2, qdd[0], qdd[1])

    f_dyn = ca.Function('f', [x_sym, u_sym], [xdot])

    # Total energy
    energy = 0.5 * x_sym[2:4].T @ M_ca @ x_sym[2:4] # kinetic energy
    y1 = - r[0] * ca.cos(q1)
    y2 = - l[0] * ca.cos(q1) - r[1] * ca.cos(q1 + q2)
    energy += m[0] * g * y1 + m[1] * g * y2

    f_energy = ca.Function('f', [x_sym], [energy])
    target_energy = f_energy(xf)
    print(f"target energy: {target_energy}")

    # Pre-built RK4 step as a CasADi Function
    Xp = ca.SX.sym('Xp', 4)
    Up = ca.SX.sym('Up', 2)

    k1 = f_dyn(Xp, Up)
    k2 = f_dyn(Xp + 0.5*dt*k1, Up)
    k3 = f_dyn(Xp + 0.5*dt*k2, Up)
    k4 = f_dyn(Xp + dt*k3, Up)
    F_step = ca.Function('F_step', [Xp, Up], [Xp + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)])

    # NLP setup 
    opti  = ca.Opti()
    X_var = opti.variable(4, N + 1)
    U_var = opti.variable(2, N)

    # OBJECTIVE FUNCTION
    # Objective: minimum control effort
    obj = 0
    # obj += ca.sumsqr(U_var)
    obj += 8*ca.sumsqr(U_var)
    obj += 0.01* ca.sumsqr(X_var - xf)
    for i in range(N):
        obj += (f_energy(X_var[:, i]) - target_energy)**2
    opti.minimize(obj)
    # opti.minimize(0.1* ca.sumsqr(X_var) + ca.sumsqr(U_var))

    # Boundary constraints
    opti.subject_to(X_var[:, 0] == x0)
    opti.subject_to(X_var[:, N] == xf)

    # Input constraints
    opti.subject_to(opti.bounded(-np.array(tau_max), U_var, np.array(tau_max)))

    # State constraints
    max_vel = np.array([10, 10])
    for i in range(N+1):
        opti.subject_to(opti.bounded(-max_vel, X_var[2:4, i], max_vel))

    # Dynamics constraints (multiple shooting, RK4)
    for k in range(N):
        opti.subject_to(X_var[:, k + 1] == F_step(X_var[:, k], U_var[:, k]))

    # Warm start: smooth position profile + inverse dynamics
    plant = DoublePendulumPlant(
        mass=mass, length=length, com=com, gravity=gravity,
        inertia=inertia, damping=damping,
        torque_limit=[np.inf, np.inf],
    )

    t_ws    = np.linspace(0, t_final, N + 1)
    # q1_ws   = np.pi * 0.5 * (np.tanh(10*t_ws/t_final - 5) + 1)
    q1_ws = np.linspace(x0[0], xf[0], N+1, axis=0)
    # breakpoint()
    dq1_ws  = np.gradient(q1_ws,  dt)
    ddq1_ws = np.gradient(dq1_ws, dt)

    X_ws = np.vstack([q1_ws, np.zeros(N + 1), dq1_ws, np.zeros(N + 1)])
    U_ws = np.zeros((2, N))
    # for k in range(N):
    #     U_ws[:, k] = plant.inverse_dynamics(X_ws[:, k], np.array([ddq1_ws[k], 0.0]))

    opti.set_initial(X_var, X_ws)
    opti.set_initial(U_var, U_ws)

    opti.solver('ipopt', {
        'ipopt.max_iter':    3000,
        'ipopt.tol':         1e-6,
        'ipopt.print_level': 5,
        "ipopt.constr_viol_tol": 0.1,
    })

    print("Solving trajectory optimization ...")
    try:
        sol = opti.solve()
    except:
        breakpoint()
    X_opt = sol.value(X_var)   # (4, N+1)
    U_opt = sol.value(U_var)   # (2, N)
    print("Solved successfully.")

    # Accelerations from forward dynamics 
    A_opt = np.zeros((2, N + 1))
    for k in range(N):
        A_opt[:, k] = plant.forward_dynamics(X_opt[:, k], U_opt[:, k])
    # Final point: zero torque (no control applied after the horizon)
    A_opt[:, -1] = plant.forward_dynamics(X_opt[:, -1], np.zeros(2))

    # Save CSV
    t_grid = np.linspace(0, t_final, N + 1)
    U_out  = np.hstack([U_opt, np.zeros((2, 1))])   # append zero for final step

    t_grid, X_opt, U_out = upsample_trajectory(t_grid, X_opt, U_out, 0.01)

    # Recompute accelerations on the upsampled grid
    N_up = t_grid.shape[0] - 1
    A_opt = np.zeros((2, N_up + 1))
    for k in range(N_up):
        A_opt[:, k] = plant.forward_dynamics(X_opt[:, k], U_out[:, k])
    A_opt[:, -1] = plant.forward_dynamics(X_opt[:, -1], np.zeros(2))

    out = np.vstack([
        t_grid,
        X_opt[0],   # pos1
        X_opt[1],   # pos2
        X_opt[2],   # vel1
        X_opt[3],   # vel2
        A_opt[0],   # acc1
        A_opt[1],   # acc2
        U_out[0],   # tau0
        U_out[1],   # tau1
    ]).T

    np.savetxt(
        csvfile,
        out,
        delimiter=",",
        header="time,pos1,pos2,vel1,vel2,acc1,acc2,tau0,tau1",
        comments="",
    )
    print(f"Saved {csvfile} ({N + 1} rows, dt={dt} s)")

    return t_grid, out[:, 1:-2].T, U_out # T, X, U

# model 06
# mass = [ 0.10548177618443695, 0.07619744360415454 ]
# length=[0.05, 0.05]
# com = [ 0.05, 0.03670036749567022 ]
# damping = [ 7.634058385430087e-12,  0.0005106535523065844 ]
# gravity=9.81
# inertia = [ 0.00046166221821039165, 0.00023702395072092597 ]
# # tau_max = np.array([0.15, 0.15])
# tau_max = np.array([0.12, 0.12])

mass = [0.10548177618443695, 0.07619744360415454]
length = [0.05, 0.05]
com = [0.05, 0.03670036749567022]
inertia = [0.00046166221821039165, 0.00023702395072092597]
damping = [7.634058385430087e-12, 0.0005106535523065844]
tau_max = [0.14, 0.14]
gravity = 9.81

# mass=[0.0654, 0.120]
# length=[0.05, 0.05]
# com = [0.275, 0.166]
# damping=[0.00018, 0.00042]
# gravity=9.81
# inertia=[0.000156, 0.000163]
# tau_max = np.array([0.15, 0.15])
dt = 0.01
tf = 5.0
csvfile = "trajectory_modelbased.csv"

T, X, U = trajectory_optimization(mass, length, com, inertia, damping, gravity, tau_max, dt, tf, csvfile)

plt.figure(figsize=(10, 6))

for i in range(6):
    plt.subplot(3, 2, i+1)
    plt.plot(T, X[i])

plt.figure(figsize=(10, 2))

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.plot(T, U[i])

plt.show()