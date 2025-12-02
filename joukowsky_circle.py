import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import seaborn as sns

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")
palette = sns.color_palette("colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 16,      # increase axis label size
    "axes.titlesize": 18,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# Joukowsky Airfoil Parameters
c = 1.0  # circle radius in z-plane
offset = 0.2-0.5j  # offset of circle center from origin (creates camber and thickness)
beta = np.arcsin(offset.imag / c)
# print(beta)
# Create a circle centered at (-offset, 0) in the z-plane
# This offset creates the characteristic airfoil shape
thetas = np.linspace(0, 2*np.pi, 1000)
z_circle = c * np.exp(1j*thetas) - offset

# Apply Joukowsky transformation: w = z + 1/z
# This maps the circle to an airfoil
w_airfoil = z_circle + (c - np.abs(offset))/z_circle

# Calculate the points on the circle
z1 = np.sqrt(c**2 - offset.imag**2) - offset.real
z2 = -np.sqrt(c**2 - offset.imag**2) - offset.real

# Apply Joukowsky transformation to these points
w1 = z1 + (c - np.abs(offset))/z1
w2 = z2 + (c - np.abs(offset))/z2

# Create figure with better size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

leading_edge_color = palette[2]
trailing_edge_color = palette[9]

# Plot 1: The circle in z-plane (showing transformation input)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.plot(z_circle.real, z_circle.imag, 'b-', linewidth=4)
ax1.plot(z2.real, z2.imag, marker='o', color=leading_edge_color, label='Leading Edge', markersize=12)
ax1.plot(z1.real, z1.imag, marker='o', color=trailing_edge_color, markersize=12, label='Trailing Edge')
# ax1.plot(-offset.real, -offset.imag, 'kx', label='Center of Circle')
ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
ax1.set_title('Off-Center Circle in z-plane')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.legend()

# Plot 2: The airfoil in w-plane (after transformation)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.plot(w_airfoil.real, w_airfoil.imag, 'r-', linewidth=4)
# ax2.fill(w_airfoil.real, w_airfoil.imag, alpha=0.2, color='red')
ax2.plot(w2.real, w2.imag, marker='o', color=leading_edge_color, label='Leading Edge', markersize=12)
ax2.plot(w1.real, w1.imag, marker='o', color=trailing_edge_color, label='Trailing Edge', markersize=12)
ax2.set_xlabel('Real')
ax2.set_ylabel('Imaginary')
ax2.set_title('Joukowsky Transformation on off-center Circle')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')
ax2.legend()

plt.tight_layout()
plt.savefig("output/joukowsky_airfoil.svg")