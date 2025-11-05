#!/usr/bin/env python
"""
FEniCS program: Phase-field model of lithium dendrite growth.
  Plotting script

  Based on the article by Taghavian, et. al.
  https://doi.org/10.1038/s41524-025-01735-x


------------------------------------------------------------------
  DrtSinX98                                                2025
------------------------------------------------------------------
"""

import matplotlib
matplotlib.use('Agg')

from fenics import *
import matplotlib.pyplot as plt
import os


"""
Main Plotting Setup
"""

# Recreate the mesh and function space to load the HDF5 data into.
# This MUST match the simulation parameters.
lox= 200
loy= 100
nx, ny = 400, 200

# Create a SEQUENTIAL mesh
mesh = RectangleMesh(Point(0, 0), Point(lox, loy), nx, ny)
P1 = FiniteElement('P', triangle, 1)
V = FunctionSpace(mesh, MixedElement([P1,P1,P1]))

u = Function(V)


"""
Define Files to Plot
"""

# For num_envsteps=2750, this corresponds to:
# n=0, n=1375, n=2749
files_to_plot = {
    'start': 'u0.h5',
    'middle': 'u1375.h5',
    'end': 'u2749.h5'
}

# Check if the files exist
for key, path in files_to_plot.items():
    if not os.path.exists(path):
        print(f"Error: Required file '{path}' not found.")
        print("Please run the simulation first.")
        print("Note: This script assumes num_envsteps=2750.")
        exit()

print("Found all required .h5 files. Starting plot...")


"""
Plotting Loop
"""

plt.figure(1)
num_plot = 0

for key, file_path in files_to_plot.items():
    print(f"Loading {file_path} for {key} plot...")
    
    # Open HDF5 file
    infile = HDF5File(mesh.mpi_comm(), file_path, "r")
    
    # Load the "solution" dataset into our function 'u'
    infile.read(u, "solution")
    infile.close()

    # Split the solution
    xi_t, w_t, phi_t = u.split()

    # Plot xi_t (Top row)
    plt.subplot(3, 3, int(num_plot + 1))
    plt.colorbar(plot(xi_t))
    plt.title(f"{key} (t={int(file_path.replace('u','').replace('.h5',''))*0.04}s)")
    plt.ylabel(r'$\xi$')

    # Plot w_t (Middle row)
    plt.subplot(3, 3, int(num_plot + 4))
    plt.colorbar(plot(w_t))
    plt.ylabel(r'$w$')

    # Plot phi_t (Bottom row)
    plt.subplot(3, 3, int(num_plot + 7))
    plt.colorbar(plot(phi_t))
    plt.ylabel(r'$\phi$')
    plt.xlabel(r'$x$')
    
    num_plot += 1


"""
Save Final Output
"""

plt.tight_layout()
plt.savefig(f'pf_sim.png')
print("Saved plot to pf_sim.png")