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

c = 1.0  # circle radius in z-plane
offset = 0.2-0.5j  # offset of circle center from origin (creates camber and thickness)
beta = np.arcsin(offset.imag / c)
# print(beta)
# Create a circle centered at (-offset, 0) in the z-plane
# This offset creates the characteristic airfoil shape
thetas = np.linspace(0, 2*np.pi, 1000)
z_circle = c * np.exp(1j*thetas) - offset

z1 = np.sqrt(c**2 - offset.imag**2) - offset.real
z2 = -np.sqrt(c**2 - offset.imag**2) - offset.real

plt.figure(figsize=(5,5))

leading_edge_color = palette[2]
trailing_edge_color = palette[9]

plt.axvline(x=0, color='k', linewidth=0.5)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.plot(z_circle.real, z_circle.imag, 'b-', linewidth=6)
plt.plot([-offset.real, z1.real], [-offset.imag, z1.imag], 'k--', linewidth=5)
plt.scatter(z2.real, z2.imag, color=leading_edge_color, marker='o',zorder=10, linewidth = 10, label='Leading Edge')
plt.scatter(z1.real, z1.imag, color=trailing_edge_color, marker='o',zorder=11, linewidth = 10, label='Trailing Edge')
plt.scatter(-offset.real, -offset.imag, color='k', marker='x', s=140, label='Center')

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Off-Center Circle in z-plane', fontsize=20)
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Move legend to not overlap with points
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("output/complex_circle_z_plane.svg")