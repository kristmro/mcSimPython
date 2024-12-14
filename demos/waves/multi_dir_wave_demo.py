

# Keep existing golden ratio parameters
# Then replace the plotting code with:

# Multidirectional wave field demo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
import os

cwd = os.getcwd()
sys.path.insert(0, cwd)

print(cwd)
print(sys.path)

from MCSimPython.waves.wave_spectra import DirectionalSpectrum, JONSWAP

# Set plot parameters
width = 426.8       # Latex document width in pts
inch_pr_pt = 1/72.27        # Ratio between pts and inches

golden_ratio = (np.sqrt(5) - 1)/2
fig_width = width*inch_pr_pt
fig_height = fig_width*golden_ratio
fig_size = [fig_width, fig_height]

params = {'backend': 'PS',
          'axes.labelsize': 10,
          'font.size': 10,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'text.usetex': True,
          'figure.figsize': fig_size} 

plt.rcParams.update(params)

# Set sea-state parameters
hs = 2.5
tp = 9.
gamma = 3.3
theta_p = 0     # Dominant wave direction

# Frequencies of wave spectra
wp = 2*np.pi/tp
wmin = .6*wp
wmax = 2.5*wp
N = 200         # Number of wave components

w = np.linspace(wmin, wmax, N)

# Generate 1D wave spectrum

jonswap = JONSWAP(w)
freq, spectrum = jonswap(hs=hs, tp=tp, gamma=gamma)

# Angles in directional spreading function 
N_theta = 100
angles = np.linspace(-np.pi, np.pi, N_theta)
s = 1       # Peakness of spreading function

rand_seed = 12345   # Random seed

dir_spectrum = DirectionalSpectrum(freq, angles, theta_p, spectrum, s=s, seed=rand_seed)
FREQ, THETA, spectrum_2d = dir_spectrum.spectrum2d()


spherical = True
if spherical:
    X1, Y1 = FREQ*np.cos(THETA), FREQ*np.sin(THETA)
else:
    X1, Y1 = FREQ, THETA
# Create a full wave field for a mesh grid
Nx = 100    # Number of discrete x locations
Ny = 100    # Number of discrete y locations

x = np.linspace(-100, 100, Nx)
y = np.linspace(-100, 100, Ny)

X, Y, wave_elevation = dir_spectrum.wave_realization(0, x=x, y=y)

# Keep existing golden ratio parameters
# Then replace the plotting code with:
# Update matplotlib backend
import matplotlib
matplotlib.use('TkAgg')  # Change to interactive backend

# First figure - Directional Spectrum
fig1 = plt.figure(figsize=fig_size)
ax1 = plt.subplot(111, projection='polar')
spectrum_plot = ax1.pcolormesh(THETA, FREQ, spectrum_2d, cmap='viridis', shading='auto')

# Set compass orientation
ax1.set_theta_zero_location('N')  # 0 degrees at North
ax1.set_theta_direction(-1)       # Clockwise direction

# Add compass labels
ax1.set_xticks(np.pi/180. * np.linspace(0, 360, 8, endpoint=False))
ax1.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_title('Directional Wave Spectrum', pad=20)
cbar = plt.colorbar(spectrum_plot)
cbar.set_label('Spectral density [m²/rad]', rotation=90, labelpad=15)
plt.tight_layout()
plt.savefig("/home/kristmro/workspace/Images/wave_spectrum_polar.eps", format='eps', dpi=300, bbox_inches='tight')
plt.show()

# Second figure - Wave Elevation
fig2 = plt.figure(figsize=fig_size)
ax2 = plt.subplot(111, projection='3d')
ax2.view_init(elev=25, azim=45)  # Fixed parameter name
wave_plot = ax2.plot_surface(X, Y, wave_elevation, 
                           cmap='coolwarm',
                           linewidth=0.5,
                           antialiased=True,
                           alpha=0.8)
ax2.set_zlim(-1.5*hs, 1.5*hs)
ax2.set_xlabel('x [m]', labelpad=10)
ax2.set_ylabel('y [m]', labelpad=10)
ax2.set_zlabel('Wave elevation [m]', labelpad=10)
ax2.grid(True, linestyle='--', alpha=0.3)

# Colorbar
cbar = plt.colorbar(wave_plot, shrink=0.8, aspect=20)
cbar.set_label('Wave elevation [m]', rotation=90, labelpad=15)

plt.title('Wave Field Realization', pad=20)
plt.tight_layout()
plt.savefig("/home/kristmro/workspace/Images/wave_elevation_3d.eps", format='eps', dpi=300, bbox_inches='tight')
plt.show()