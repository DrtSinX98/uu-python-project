"""
FEniCS program: Phase-field model of lithium dendrite growth.
  Post-processing script - Surface Roughness & Charge

  Based on the article by Taghavian, et. al.
  https://doi.org/10.1038/s41524-025-01735-x


------------------------------------------------------------------
  DrtSinX98                                                2025
------------------------------------------------------------------
"""


from fenics import *
from boxfield import *
import numpy as np
import scipy.io
import glob
from datetime import datetime


"""
Calculates the amount of charge and the surface roughness.
"""

def CVD(u):
    # Mesh dimension and size
    lox= 200
    loy= 100
    nx, ny = 400, 200 
    
    Csm=7.64e4**(2/3) # unit: mol/m2. Converted to 2D.
    n=1 #charge valence
    F=9.648533e4 # unit: C/mol. Faraday constant
    
    xi_t, w_t, phi_t = u.split(deepcopy=True)
    u_box = FEniCSBoxField(xi_t, (nx, ny))
    
    x = u_box.grid.coor[X]
    delx = x[1] - x[0]
    y = u_box.grid.coor[Y]
    dely = y[1] - y[0]
    delxy=delx*dely*1e-12 # unit:m^2   
    
    
    Usumz=np.zeros((1,u_box.values.shape[1]))
    Usumzh=np.zeros((1,u_box.values.shape[1]))
    
    # Iterate over 2D mesh points (i, j)
    for j in range(u_box.values.shape[1]): # Y direction
        #using zeta
        Usumz[0,j]=np.sum(u_box.values[:,j])*delx*1e-6 # unit: m
        #using the function h(zeta):
        hzeta= np.power(u_box.values[:,j],3)*(6.0*np.power(u_box.values[:,j],2) -15.0*u_box.values[:,j] + 10.0)
        Usumzh[0,j]=np.sum(hzeta*Csm*delxy)
    
    # charge
    chamol=np.sum(Usumzh) # unit: mol
    cha=chamol*n*F # unit: C
    
    # valley level
    val=np.min(Usumz) # unit: m
    
    # dendrite level    
    den=np.max(Usumz) # unit: m
    
    return [cha,val,den]



"""
Main Post-Processing Setup
"""

# Recreate the mesh and function space to load the HDF5 data into.
# This MUST match the simulation parameters.
lox= 200
loy= 100
nx, ny = 400, 200
Tfin = 0.04

# Create a SEQUENTIAL mesh
mesh = RectangleMesh(Point(0, 0), Point(lox, loy), nx, ny)
P1 = FiniteElement('P', triangle, 1)
V = FunctionSpace(mesh, MixedElement([P1,P1,P1]))

u = Function(V)

# Find all saved HDF5 files
file_list = glob.glob('u*.h5')
# Sort by the integer number in the filename.
file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

num_steps = len(file_list)

if num_steps == 0:
    print("No u*.h5 files found. Run the simulation first.")
    exit()

print(f"Found {num_steps} HDF5 files to process: {file_list}")

# Create arrays to store results
Rough = np.zeros((1, num_steps))
Charge = np.zeros((1, num_steps))
# Re-create the time axis based on the step number in the filename
Taxis = np.array([int(''.join(filter(str.isdigit, f))) for f in file_list]) * Tfin



"""
Processing Loop
"""

for i, file_path in enumerate(file_list):
    print(f"Processing {file_path} (Time: {Taxis[i]}s)...")
    
    # Open HDF5 file
    infile = HDF5File(mesh.mpi_comm(), file_path, "r")
    
    # Load the "solution" dataset into our function 'u'
    infile.read(u, "solution")
    infile.close()
    
    [cha, val, den] = CVD(u)
    
    Rough[0, i] = den - val
    Charge[0, i] = cha

print("Post-processing complete.")



"""
Save Final Output
"""

Raxis = Rough[0, :]
Chaxis = Charge[0, :]
matdic = {"roughness": Raxis, "charge": Chaxis, "time": Taxis}

scipy.io.savemat('Outputs.mat', matdic)
print(f"Saved all results to Outputs.mat")