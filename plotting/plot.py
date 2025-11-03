"""
FEniCS program: Phase-field model of lithium dendrite growth.
  Plot solutions.

------------------------------------------------------------------
  rbkrgb                                                   2020
------------------------------------------------------------------
"""

from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from boxfield import *


"""
SELECT SIMULATION
"""
n_sim = 0      # 0..2



"""
PARAMETERS
"""

# Mesh dimension and size
lox= 200
loy= 100
nx, ny = 400, 200

# Normalized parameters
L = Constant(6.25)
kappa = Constant(0.3) # =R*T
Ls = Constant(0.001)
alpha = Constant(0.5)
AA = Constant(38.69)   # nF/RT
W = Constant(2.4)
es = Constant(-13.8)
el = Constant(2.631)  # mu0 = -R*T*ln(c0),, el = ln(cl)
A = Constant(1.0) # =R*T/R*T
c0 = Constant(1.0/14.89)
dv = Constant(5.5)   # Csm/Clm
M0 = Constant(317.9)
S1 = Constant(1000000)
S2 = Constant(1.19)
ft2 = Constant(0.0074)



"""
MESH AND  INITIAL CONDITION
"""

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(lox, loy), nx, ny)
P1 = FiniteElement('P', triangle, 1)
V = FunctionSpace(mesh, MixedElement([P1,P1,P1]))

# Load initial condition
fsolution=f"saved/sim{n_sim+1}/u_t0.xml"
u1=Function(V, fsolution)



"""
PHI AND C_+ PROFILES IN VALLEY AND DENDRITE REGIONS
"""

# Profiles

c0 = 1.0/14.89

tiempo2={
    0: [0,60,100,105,108],
    1: [0,60,100,105,108],
    2: [0,60,80,100,105]
}
t = tiempo2.get(n_sim, 0)

y_valle = [ 70, 76, 27, 18 ]
y_dend = [ 100, 100, 100, 38 ]

u2 = Function(V, f"saved/sim{n_sim+1}/u_t{t[1]}.xml")
u3 = Function(V, f"saved/sim{n_sim+1}/u_t{t[2]}.xml")
u4 = Function(V, f"saved/sim{n_sim+1}/u_t{t[3]}.xml")
u5 = Function(V, f"saved/sim{n_sim+1}/u_t{t[4]}.xml")

xi_1, w_1, phi_1 = u1.split(deepcopy=True)
xi_2, w_2, phi_2 = u2.split(deepcopy=True)
xi_3, w_3, phi_3 = u3.split(deepcopy=True)
xi_4, w_4, phi_4 = u4.split(deepcopy=True)
xi_5, w_5, phi_5 = u5.split(deepcopy=True)


#  phi
fig = plt.figure(figsize=(9,3))
fig.subplots_adjust(bottom=0.18, wspace=0.45)

#    Valley
plt.subplot(1, 2, 1)
start = (0, y_valle[n_sim])
T_box = FEniCSBoxField(phi_1, (nx, ny))
x, phi_val1, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_2, (nx, ny))
x, phi_val2, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_3, (nx, ny))
x, phi_val3, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_4, (nx, ny))
x, phi_val4, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_5, (nx, ny))
x, phi_val5, y_fixed, snapped=T_box.gridline(start, direction=X)

plt.plot(x, phi_val1, x, phi_val2, x, phi_val3, x, phi_val4, x, phi_val5)
plt.title(f'Valley region (y={y_valle[n_sim]}$\mu m$)')
plt.xlabel('$x \; (\mu m)$')
plt.ylabel('$\\phi \; (V)$')
plt.ylim([-0.5, 0.1])
plt.legend([f't = {t[0]}',f't = {t[1]}',f't = {t[2]}',f't = {t[3]}',f't = {t[4]}'], loc='upper left', fontsize='x-small')

#    Dendrite
plt.subplot(1, 2, 2)
start = (0, y_dend[n_sim])
T_box = FEniCSBoxField(phi_1, (nx, ny))
x, phi_val1, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_2, (nx, ny))
x, phi_val2, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_3, (nx, ny))
x, phi_val3, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_4, (nx, ny))
x, phi_val4, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(phi_5, (nx, ny))
x, phi_val5, y_fixed, snapped=T_box.gridline(start, direction=X)

plt.plot(x, phi_val1, x, phi_val2, x, phi_val3, x, phi_val4, x, phi_val5)
plt.title(f'Dendrite region (y={y_dend[n_sim]}$\mu m$)')
plt.xlabel('$x \; (\mu m)$')
plt.ylabel('$\\phi \; (V)$')
plt.ylim([-0.5, 0.1])
plt.legend([f't = {t[0]}',f't = {t[1]}',f't = {t[2]}',f't = {t[3]}',f't = {t[4]}'], loc='upper left', fontsize='x-small')

# Save plots in file
#plt.savefig(f'saved/images/DEND_TEST{n_sim}_profiles_phi.png')


#  c
fig = plt.figure(figsize=(9,3))
fig.subplots_adjust(bottom=0.18, wspace=0.45)

#    Valley
plt.subplot(1, 2, 1)
start = (0, y_valle[n_sim])
T_box = FEniCSBoxField(w_1, (nx, ny))
x, w_val1, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_2, (nx, ny))
x, w_val2, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_3, (nx, ny))
x, w_val3, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_4, (nx, ny))
x, w_val4, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_5, (nx, ny))
x, w_val5, y_fixed, snapped=T_box.gridline(start, direction=X)

plt.plot(x, c0*np.exp(w_val1), x, c0*np.exp(w_val2), x, c0*np.exp(w_val3), x, c0*np.exp(w_val4), x, c0*np.exp(w_val5))
plt.title(f'Valley region (y={y_valle[n_sim]}$\mu m$)')
plt.xlabel('$x \; (\mu m)$')
plt.ylabel('$c_+ \; (V)$')
plt.ylim([0., 0.15])
plt.legend([f't = {t[0]}',f't = {t[1]}',f't = {t[2]}',f't = {t[3]}',f't = {t[4]}'], loc='upper left', fontsize='x-small')

#    Dendrite
plt.subplot(1, 2, 2)
start = (0, y_dend[n_sim])
T_box = FEniCSBoxField(w_1, (nx, ny))
x, w_val1, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_2, (nx, ny))
x, w_val2, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_3, (nx, ny))
x, w_val3, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_4, (nx, ny))
x, w_val4, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(w_5, (nx, ny))
x, w_val5, y_fixed, snapped=T_box.gridline(start, direction=X)

plt.plot(x, c0*np.exp(w_val1), x, c0*np.exp(w_val2), x, c0*np.exp(w_val3), x, c0*np.exp(w_val4), x, c0*np.exp(w_val5))
plt.title(f'Dendrite region (y={y_dend[n_sim]}$\mu m$)')
plt.xlabel('$x \; (\mu m)$')
plt.ylabel('$c_+ \; (V)$')
plt.ylim([0., 0.15])
plt.legend([f't = {t[0]}',f't = {t[1]}',f't = {t[2]}',f't = {t[3]}',f't = {t[4]}'], loc='upper left', fontsize='x-small')

# Save plots in file
#plt.savefig(f'saved/images/DEND_TEST{n_sim}_profiles_c.png')



"""
ALL PROFILES IN DENDRITE REGION
"""

# All profiles
fig = plt.figure(figsize=(8,6))

start = (0, y_dend[n_sim])
T_box = FEniCSBoxField(xi_1, (nx, ny))
x, xi_val1, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(xi_2, (nx, ny))
x, xi_val2, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(xi_3, (nx, ny))
x, xi_val3, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(xi_4, (nx, ny))
x, xi_val4, y_fixed, snapped=T_box.gridline(start, direction=X)
T_box = FEniCSBoxField(xi_5, (nx, ny))
x, xi_val5, y_fixed, snapped=T_box.gridline(start, direction=X)

plt.plot(x, xi_val1, x, xi_val2, x, xi_val3, x, xi_val4, x, xi_val5, x, c0*np.exp(w_val1), x, c0*np.exp(w_val2), x, c0*np.exp(w_val3), x, c0*np.exp(w_val4), x, c0*np.exp(w_val5), x, phi_val1+0.45, x, phi_val2+0.45, x, phi_val3+0.45, x, phi_val4+0.45, x, phi_val5+0.45)
plt.title(f'$\\xi$, $c_+$ and $\\phi$ in dendrite region (y={y_dend[n_sim]}$\mu m$)')
plt.xlabel('$x \; (\mu m)$')


# Show plots
plt.show()
