import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as mplanimation
try:
    from pycloudpendulum import Client
except:
    from cloud_pendulum_local import Client
import os
import time
from misc import download_video

class PendulumPlant:
    def __init__(self, mass=1.0, length=0.5, damping=0.1, gravity=9.81, inertia=None, torque_limit=np.inf):
        self.m = mass
        self.l = length
        self.b = damping
        self.g = gravity
        if inertia is None:
            self.I = mass*length*length
        else:
            self.I = inertia
        self.torque_limit = torque_limit

        self.dof = 1
        self.x = np.zeros(2*self.dof) #position, velocity
        self.t = 0.0 #time

        self.t_values = []
        self.x_values = []
        self.tau_values = []

    def set_state(self, time, x):
        self.x = x
        self.t = time

    def get_state(self):
        return self.t, self.x

    def forward_kinematics(self, pos):
        """
        forward kinematics, origin at fixed point
        """
        ee_pos_x = self.l * np.sin(pos)
        ee_pos_y = -self.l*np.cos(pos)
        return [ee_pos_x, ee_pos_y]

    def forward_dynamics(self, pos, vel, tau):
        """
        return acceleration from current position, velocity and torque.
        use self.m, self.g, self.l, self.b and self.I if needed
        """
        torque = np.clip(tau, -self.torque_limit, self.torque_limit)

        accn = (torque - self.m*self.g*self.l*np.sin(pos) - self.b*vel) / self.I
        
        return accn

    def inverse_dynamics(self, pos, vel, accn):
        """
        return torque acting on the revolute joint (tau) in terms of inputs
        use self.m, self.g, self.l, self.b and self.I if needed
        """
        tau_id = accn*self.I + self.m*self.g*self.l*np.sin(pos) + self.b*vel

        return tau_id

    def rhs(self, t, x, tau):

        """
        Computes the integrand of the equations of motion.
        """
        accn = self.forward_dynamics(x[0], x[1], tau)
        integ = np.array([x[1], accn])
        return integ

    def euler_integrator(self, t, y, dt, tau):
        """
        Implement Forward Euler Integration for a time-step dt and state y
        y = [pos, vel]
        """
        integ = self.rhs(t, y, tau)
        y_new = y + dt*integ
        return y_new

    def runge_integrator(self, t, y, dt, tau):
        """
        Bonus: Implement a fourth order Runge-Kutta Integration scheme
        """
        k1 = self.rhs(t, y, tau)
        k2 = self.rhs(t + 0.5*dt, y + 0.5*dt*k1, tau)
        k3 = self.rhs(t + 0.5*dt, y + 0.5*dt*k2, tau)
        k4 = self.rhs(t + dt, y + dt*k3, tau)
        integ = (k1 + 2*(k2 + k3) + k4) / 6.0

        y_new = y + dt*integ

        return y_new

    def step(self, tau, dt, integrator="euler"):
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        if integrator == "runge_kutta":
            self.x = self.runge_integrator(self.t, self.x, dt, tau)
        elif integrator == "euler":
            self.x = self.euler_integrator(self.t, self.x, dt, tau)
        self.t += dt
        # Store the time series output
        #if len(self.x_values) == 500:
        #    self.x = self.x - [np.pi, 0]
        self.t_values.append(self.t)
        self.x_values.append(self.x.copy())
        self.tau_values.append(tau)

    def simulate(self, t0, y0, tf, dt, controller=None, integrator="euler"):
        self.set_state(t0, y0)
        
        self.t_values = []
        self.x_values = []
        self.tau_values = []
        self.controller = controller
        
        while (self.t <= tf):
            if controller is not None:
                #tau = controller.get_control_output(self.x, self.t)
                tau = controller.get_control_output(self.x)
            else:
                tau = 0
            self.step(tau, dt, integrator=integrator)

        return self.t_values, self.x_values, self.tau_values
    
    def plot_MPC_instant_trajectory(self, x_trj, u_trj, dt, N, i):
        # Plot the trajectory which the system has followed so far
        # as well as the trajectory which has been calculated
        # until the horizon (th)
        ## Executed trajectory up to current point
        plt.plot(self.t_values[:i], np.asarray(self.x_values[:i]).T[0], color = "tab:blue", label="theta")
        plt.plot(self.t_values[:i], np.asarray(self.x_values[:i]).T[1], color = "tab:orange", label="theta dot")
        plt.plot(self.t_values[:i], self.tau_values[:i], color = "tab:green", label="u")
        ## Currently planned trajectory
        self.t_values_th = np.linspace(self.t_values[i], self.t_values[i]+dt*(N+1), N)
        plt.plot(self.t_values_th, np.asarray(x_trj).T[0], color = "tab:blue", linestyle = "dashed", label="theta")
        plt.plot(self.t_values_th, np.asarray(x_trj).T[1], color = "tab:orange", linestyle = "dashed", label="theta dot")
        plt.plot(self.t_values_th[:-1], u_trj, color = "tab:green", linestyle = "dashed", label="u")
        plt.legend(loc="best")
        
        #plt.show()
        
    def animate_MPC_plot2(self, x_trj, u_trj, cost, ite, dt, N, vid = False, vidname = "video"):
        fig, (ax1, ax3) = plt.subplots(2)
        ax2 = ax1.twinx()
        ax4 = ax3.twinx()
        
        ## Executed trajectory up to current point
        x1 = ax1.plot(0, 0, color = "tab:blue", label="theta")[0]
        x2 = ax1.plot(0, 0, color = "tab:orange", label="theta dot")[0]
        u = ax2.plot(0, 0, color = "tab:green", label="u")[0]
        
        ## Currently planned trajectory
        
        x1traj = ax1.plot(0, 0, color = "tab:blue", linestyle = "dashed", label="theta")[0]
        x2traj = ax1.plot(0, 0, color = "tab:orange", linestyle = "dashed", label="theta dot")[0]
        utraj = ax2.plot(0, 0, color = "tab:green", linestyle = "dashed", label="u")[0]

        ax1.set(xlim=[0, 10], ylim=[-7.5, 7.5], xlabel='Time [s]')
        ax1.legend(loc="best")
        ax2.legend(loc="best")

        ax2.set(ylim=[-0.1, 0.1])
        ax2.set_ylabel(ylabel='u [Nm]', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        ## iterations

        costpl = ax3.plot(0, 0, color = "tab:blue", label = "cost")[0]
        itepl = ax4.plot(0, 0, color = "tab:orange", label = "iterations")[0]

        ax3.set(xlim=[0, 10], ylim=[0.001, 100], xlabel='Time [s]', ylabel='Cost')
        ax4.set(ylim = [0, 50])
        ax4.set_ylabel(ylabel='Iterations', color='tab:orange')
        ax4.tick_params(axis='y', labelcolor='tab:orange')
        ax3.set_yscale('log')
        
        fig.tight_layout()
        
        def update_MPC_plot(frame):
            frame = frame*10
            
            x1.set_xdata(self.t_values[:frame])
            x1.set_ydata(np.asarray(self.x_values).T[0][:frame])

            x2.set_xdata(self.t_values[:frame])
            x2.set_ydata(np.asarray(self.x_values).T[1][:frame])

            u.set_xdata(self.t_values[:frame])
            u.set_ydata(self.tau_values[:frame])

            t2 = np.linspace(self.t_values[frame], self.t_values[frame]+dt*(N+1), N)
            
            x_trji = np.asarray([x_trj[frame*2], x_trj[frame*2+1]]).T
            u_trji = np.asarray(u_trj)[frame]
            
            x1traj.set_xdata(t2)
            x1traj.set_ydata(np.asarray(x_trji).T[0])

            x2traj.set_xdata(t2)
            x2traj.set_ydata(np.asarray(x_trji).T[1])

            utraj.set_xdata(t2[:-1])
            utraj.set_ydata(u_trji)

            costpl.set_xdata(self.t_values[:frame])
            costpl.set_ydata(cost[:frame])

            itepl.set_xdata(self.t_values[:frame])
            itepl.set_ydata(ite[:frame])

            return (x1, x2, u, x1traj, x2traj, utraj)

        ani = mplanimation.FuncAnimation(fig=fig, func=update_MPC_plot, frames=int(len(self.t_values)/10), interval = 100)
        if vid:
            ani.save(vidname + ".gif")
        plt.show()    
    '''
    def animate_MPC_plot(self):
        moviewriter = mplanimation.FFMpegWriter(fps = 10)
        fig = plt.figure()
        x_trj , u_trj, dt, N = self.controller.get_animation_data()
        with moviewriter.saving(fig, "vid1.mp4", 100):
            for i in range(0, 1000, 10):
                plt.clf()
                plot_MPC_instant_trajectory(np.asarray([x_trj[i*2], x_trj[i*2+1]]).T, u_trj[i], dt, N, i)
                moviewriter.grab_frame()
        
        #for i in range(self.t_values.len())
    '''         
    def simulate_and_animate(self, t0, y0, tf, dt, controller=None, integrator="euler", save_video=False):
        """
        simulate and animate the pendulum
        """
        self.set_state(t0, y0)

        self.t_values = []
        self.x_values = []
        self.tau_values = []

        #fig = plt.figure(figsize=(6,6))
        #self.animation_ax = plt.axes()
        fig, (self.animation_ax, self.ps_ax) = plt.subplots(1, 2, figsize=(10, 5))
        self.animation_plots = []
        ee_plot, = self.animation_ax.plot([], [], "o", markersize=25.0, color="blue")
        bar_plot, = self.animation_ax.plot([], [], "-", lw=5, color="black")
        #text_plot = self.animation_ax.text(0.1, 0.1, [], xycoords="figure fraction")
        self.animation_plots.append(ee_plot)
        self.animation_plots.append(bar_plot)

        num_steps = int(tf / dt)
        par_dict = {}
        par_dict["dt"] = dt
        par_dict["controller"] = controller
        par_dict["integrator"] = integrator
        frames = num_steps*[par_dict]

        #ps_fig = plt.figure(figsize=(6,6))
        #self.ps_ax = plt.axes()
        #self.ps_plots = []
        ps_plot, = self.ps_ax.plot([], [], "-", lw=1.0, color="blue")
        #self.ps_plots.append(ps_plot)
        self.animation_plots.append(ps_plot)

        animation = FuncAnimation(fig, self._animation_step, frames=frames, init_func=self._animation_init, blit=True, repeat=False, interval=dt*1000)
        animation2 = None
        #if phase_plot:
        #    animation2 = FuncAnimation(fig, self._ps_update, init_func=self._ps_init, blit=True, repeat=False, interval=dt*1000)

        if save_video:
            Writer = mplanimation.writers['ffmpeg']
            writer = Writer(fps=60, bitrate=1800)
            mplanimation.save('pendulum_swingup.mp4', writer=writer)
            #if phase_plot:
            #    Writer2 = mplanimation.writers['ffmpeg']
            #    writer2 = Writer2(fps=60, bitrate=1800)
            #    animation2.save('pendulum_swingup_phase.mp4', writer=writer2)
        #plt.show()

        return self.t_values, self.x_values, self.tau_values, animation#, animation2

    def _animation_init(self):
        """
        init of the animation plot
        """
        self.animation_ax.set_xlim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_ylim(-1.5*self.l, 1.5*self.l)
        self.animation_ax.set_xlabel("x position [m]")
        self.animation_ax.set_ylabel("y position [m]")
        for ap in self.animation_plots:
            ap.set_data([], [])

        self._ps_init()
        return self.animation_plots

    def _animation_step(self, par_dict):
        """
        simulation of a single step which also updates the animation plot
        """
        dt = par_dict["dt"]
        controller = par_dict["controller"]
        integrator = par_dict["integrator"]
        if controller is not None:
            tau = controller.get_control_output(self.x)
        else:
            tau = 0
        self.step(tau, dt, integrator=integrator)
        ee_pos = self.forward_kinematics(self.x[0])
        self.animation_plots[0].set_data((ee_pos[0],), (ee_pos[1],))
        self.animation_plots[1].set_data([0, ee_pos[0]], [0, ee_pos[1]])

        self._ps_update(0)

        return self.animation_plots

    def _ps_init(self):
        """
        init of the phase space animation plot
        """
        self.ps_ax.set_xlim(-np.pi, 2*np.pi)
        self.ps_ax.set_ylim(-10, 10)
        self.ps_ax.set_xlabel("degree [rad]")
        self.ps_ax.set_ylabel("velocity [rad/s]")
        for ap in self.animation_plots:
            ap.set_data([], [])
        return self.animation_plots

    def _ps_update(self, i):
        """
        update of the phase space animation plot
        """
        self.animation_plots[-1].set_data(np.asarray(self.x_values).T[0], np.asarray(self.x_values).T[1])
        return self.animation_plots

    def run_on_hardware(self, user_token, x0, tf, dt, controller=None, save_video=True):
        """
        Run an experiment on the cloud pendulum (or on a local setup, if the
        cloud client is not installed).

        :param x0: initial angle of the pendulum (rad)
        :param save_video: download the recording (only possible on the cloud)
        :return: times, states (N x 2), measured torques, desired torques,
            video url, video path
        """
        client = Client()
        session_token, livestream_url = client.start_experiment(
            user_token = user_token,
            experiment_type = "SimplePendulum",
            experiment_time = tf,
            preparation_time = 5.0,
            initial_position = [x0],
            record = True
        )
        client.set_impedance_controller_params([0.0], [0.0], session_token)
        meas_time = 0.0
        meas_dt = 0.0
        n = int(tf / dt)
        i = 0
        self.t_values = np.zeros(n)
        self.x_values = np.zeros((2,n))
        self.tau_values = np.zeros(n)
        self.des_tau_values = np.zeros(n)

        while meas_time < tf and i < n:
            start_loop = time.time()
            meas_time += meas_dt

            # Measure data
            # the client returns one value per joint; this plant has a single joint
            measured_position = client.get_position(session_token)[0] # Measure position
            measured_velocity = client.get_velocity(session_token)[0] # Measure velocity
            measured_torque = client.get_torque(session_token)[0] # Measure torque

            # controller
            if controller is None:
                tau = 0.0
            else:
                tau = controller.get_control_output(np.array([measured_position, measured_velocity]))
                client.set_torque([tau], session_token)

            # add data to matrices
            self.t_values[i] = meas_time
            self.x_values[:,i] = np.array([measured_position, measured_velocity])
            self.tau_values[i] = measured_torque
            self.des_tau_values[i] = tau

            self.wait_for_control_loop_end(start_loop, dt)

            meas_dt = time.time() - start_loop
            i = i + 1

        video_url = client.stop_experiment(session_token)

        if save_video:
            video_path = download_video(video_url)
        else:
            video_path = ""

        return self.t_values[:i], self.x_values[:,:i].T, self.tau_values[:i], self.des_tau_values[:i], video_url, video_path

    def wait_for_control_loop_end(self, start_loop, dt):
        """Delay ending a while loop so that it loops at a desired sampling time dt."""
        time_to_pass = start_loop + dt - time.time()
        if time_to_pass <= 0.0:
            return

        time.sleep(time_to_pass * 0.7) # sleep
        while time.time() - start_loop < dt: # busy waiting
            pass

def plot_timeseries(T, X, U):
    plt.plot(T, np.asarray(X).T[0], label="theta")
    plt.plot(T, np.asarray(X).T[1], label="theta dot")
    plt.plot(T, U, label="u")
    plt.legend(loc="best")
    plt.show()

