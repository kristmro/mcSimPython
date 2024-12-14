
import sys
sys.path.append(sys.path)  # Adjust this path as needed
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

from src.MCSimPython.simulator.csad import CSAD_DP_6DOF
from src.MCSimPython.guidance.filter import ThrdOrderRefFilter
from src.MCSimPython.control.basic import PD, PID
from src.MCSimPython.utils import Rz, six2threeDOF, three2sixDOF
from src.MCSimPython.waves import JONSWAP, WaveLoad

plt.rcParams.update({
    'figure.figsize': (8, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.grid': True
})

# Simulation parameters
simtime = 80
dt = 0.1
t = np.arange(0, simtime, dt)

# Vessel
vessel = CSAD_DP_6DOF(dt, method="RK4")
#vessel_pid = CSAD_DP_6DOF(dt, method="RK4")
control = PD(kp=[0, 0, 0], kd=[0.0, 0.0, 0])
#control_i = PID(kp=[10, 10, 50.], kd=[100.5, 100.5, 200.], ki=[.3, .3, .4], dt=dt)
ref_model = ThrdOrderRefFilter(dt, omega=[0.2, 0.05, .02])

# Current 
uc = 0.00                   # Current velocity [m/s]
beta_c = np.deg2rad(0)      # Current direction [rad]

eta = np.zeros((len(t), 3))
nu = np.zeros((len(t), 3))
#eta_pid = np.zeros_like(eta)
#nu_pid = np.zeros_like(eta)
xd = np.zeros((len(t), 9))

# Set reference points for a fore corner test
set_points = np.zeros((len(t), 3))
distance = 3
t_1 = t > 50
t_2 = t > 150
t_3 = t > 300
t_4 = t > 400
t_5 = t > 600

set_points[t_1, 0] = distance
set_points[t_2, 0] = distance
set_points[t_2, 1] = distance
set_points[t_4, 0] = 0
set_points[t_4, 1] = distance
set_points[t_5, 1] = 0

# Wave parameters
hs = 2.5  # Significant wave height [m]
tp = 12.0  # Peak period [s]
wp = 2 * np.pi / tp

N = 100
wmin = wp / 2.
wmax = wp * 3.
dw = (wmax - wmin) / N

w = np.linspace(wmin, wmax, N, endpoint=True)
k = w ** 2 / 9.81

jonswap = JONSWAP(w)
_, spectrum = jonswap(hs, tp, gamma=3.3)

wave_amps = np.sqrt(2 * spectrum * dw)
wave_angle = -np.pi * np.ones(N)
eps = np.random.uniform(0, 2 * np.pi, size=N)

waveload = WaveLoad(
    wave_amps,
    freqs=w,
    eps=eps,
    angles=wave_angle,
    config_file=vessel._config_file,
    interpolate=True,
    qtf_method="geo-mean"
)

# Simulate response: compare PD with PID control
for i in range(1, len(t)):
    ref_model.set_eta_r(set_points[i])
    ref_model.update()
    eta_d = ref_model.get_eta_d()
    eta_d_dot = ref_model.get_eta_d_dot()
    nu_d = Rz(vessel.get_eta()[-1]).T @ eta_d_dot
    #nu_d2 = Rz(vessel_pid.get_eta()[-1]).T @ eta_d_dot
    xd[i] = ref_model._x

    tau_cmd = control.get_tau(six2threeDOF(vessel.get_eta()), eta_d, six2threeDOF(vessel.get_nu()), nu_d)
    #tau_cmd_pid = control_i.get_tau(six2threeDOF(vessel_pid.get_eta()), eta_d, six2threeDOF(vessel_pid.get_nu()), nu_d2)

    # Compute wave loads
    tau_wf = waveload.first_order_loads(t[i], vessel.get_eta())+waveload.second_order_loads(t[i], vessel.get_eta()[-1])
    #tau_wf_pid = waveload.first_order_loads(t[i], vessel_pid.get_eta())

    # Integrate vessel dynamics with wave loads
    vessel.integrate(uc, beta_c, three2sixDOF(tau_cmd) + tau_wf)
    #vessel_pid.integrate(uc, beta_c, three2sixDOF(tau_cmd_pid) + tau_wf_pid)

    eta[i] = six2threeDOF(vessel.get_eta())
    nu[i] = six2threeDOF(vessel.get_nu())
    #eta_pid[i] = six2threeDOF(vessel_pid.get_eta())
    #nu_pid[i] = six2threeDOF(vessel_pid.get_nu())

plt.figure(figsize=(6, 6))
plt.axis("equal")
plt.plot(xd[:, 1], xd[:, 0], 'r-', label="$\eta_d$")
plt.plot(eta[:, 1], eta[:, 0], 'k-', label="PD")
#plt.plot(eta_pid[:, 1], eta_pid[:, 0], 'b--', label="PID")
plt.xlabel("E [m]")
plt.ylabel("N [m]")
plt.legend()
fig, ax = plt.subplots(3, 1, sharex=True)
fig.suptitle("Vessel states")
#give title to the plots
ax[0].set_title("Surge")
ax[1].set_title("Sway")
ax[2].set_title("Yaw")
for i in range(3):
    plt.sca(ax[i])
    plt.plot(t, eta[:, i], label="PD")
    #plt.plot(t, eta_pid[:, i], label="PID")
    plt.plot(t, xd[:, i], label="$\eta_d$")
    plt.legend()
    
fig, ax = plt.subplots(3, 1, sharex=True)
fig.suptitle("Vessel velocities")
#give title to the plots
ax[0].set_title("Surge")
ax[1].set_title("Sway")
ax[2].set_title("Yaw")
for i in range(3):
    plt.sca(ax[i])
    plt.plot(t, xd[:, i+3], label=r"$\nu_d$")
    plt.plot(t, nu[:, i], 'k-', label="PD")
    #plt.plot(t, nu_pid[:, i], 'm--', label="PID")
    plt.legend()

fig, ax = plt.subplots(3, 1, sharex=True)
fig.suptitle("Control forces")
#give title to the plots
ax[0].set_title("Surge")
ax[1].set_title("Sway")
ax[2].set_title("Yaw")
for i in range(3):
    #plotting the forces
    plt.sca(ax[i])
    plt.plot(t, tau_cmd[i] * np.ones_like(t), 'k-', label="PD")
    #plt.plot(t, tau_cmd_pid[i] * np.ones_like(t), 'm--', label="PID")
    plt.legend()
    
plt.show()
#plot the NED position of the vessel
plt.figure(figsize=(6, 6))
plt.axis("equal")
plt.plot(eta[:, 1], eta[:, 0], 'k-', label="displacement")
plt.xlabel("E [m]")
plt.ylabel("N [m]")
plt.legend()
plt.show()