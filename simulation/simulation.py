"""
FEniCS program: Phase-field model of lithium dendrite growth.
  Parallel version

  Based in the article of Hong and Viswanathan:
  https://doi.org/10.1021/acsenergylett.8b01009
  Modified as per the article by Taghavian, et. al.
  https://doi.org/10.1038/s41524-025-01735-x


------------------------------------------------------------------
  rbkrgb                                                   2020
  DrtSinX98                                                2026
------------------------------------------------------------------
"""

from fenics import *
from dolfin import MPI
from datetime import datetime
import numpy as np

# Paralell settings
comm = MPI.comm_world
rank = MPI.rank(comm)

# Start message
if rank == 0:
    print("Starting simulation at ", datetime.fromtimestamp(datetime.timestamp(datetime.now())))

"""
SELECT SIMULATION  0..2
"""
n_sim = 0



"""
PARAMETERS
"""

# Mesh dimension and size
lox= 200
loy= 100
nx, ny = 400, 200

# Normalized Parameters 
L = Constant(6.25)          #L_sigma
kappa = Constant(0.3)       #kappa
Ls = Constant(0.001)        #L_eta
alpha = Constant(0.5)       #alpha
AA = Constant(38.69)        #nF/RT
W = Constant(2.4)           #W
es = Constant(-13.8)        #e^s
el = Constant(2.631)        #e^l
A = Constant(1.0)           #R*T/R*T
c0 = Constant(1.0/14.89)    #c_0
dv = Constant(5.5)         #Csm/Clm
M0 = Constant(317.9)       #D^l                               
S1 = Constant(1000000)      #sigma^s                             
S2 = Constant(1.19)        #sigma^l                        
ft2 = Constant(0.0074)      #nFCsm



"""
MESH AND FUNCTIONS
"""

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(lox, loy), nx, ny)
P1 = FiniteElement('P', triangle, 1)
V = FunctionSpace(mesh, MixedElement([P1,P1,P1]))

# Define trial and  test functions
v_1, v_2, v_3 = TestFunctions(V)

# Define functions for solutions at previous and at current time
u = Function(V)                # At current time
u_n = Function(V)

# Split system function to access the components
xi, w, phi = split(u)
xi_n, w_n, phi_n = split(u_n)



"""
INITIAL CONDITIONS
"""

# Create initial conditions
if n_sim == 2:
    ruido = Expression('0.04*sin(0.0628*(x[1]-25))', degree=3)
elif n_sim == 1:
    ruido = Expression('0.175*exp(-0.1*pow(x[1]-50.,2.0))', degree=3)
else:
    ruido = Expression('0.04*sin(x[1])', degree=3)
u_init = Expression(('0.5*(1.0-1.0*tanh((x[0]-20.0+ruido)*2))','x[0] < (20.0) ? -10.0 : 0.0','-0.225*(1.0-tanh((x[0]-20.0+ruido)*2))'), degree=3, ruido=ruido)
u.interpolate(u_init)
u_n.interpolate(u_init)



"""
BOUNDARY CONDITIONS
"""

# Define boundary conditions

# Boundaries y=0, y=Ly
def boundary0(x, on_boundary):
    return on_boundary and near(x[0], 0)
def boundaryL(x, on_boundary):
    return on_boundary and near(x[0], lox)

# Boundary conditions for xi
bc_xi1 = DirichletBC(V.sub(0), Constant(1.0), boundary0)
bc_xi2 = DirichletBC(V.sub(0), Constant(0.0), boundaryL)

# Boundary conditions for mu
bc_c2 = DirichletBC(V.sub(1), Constant(0.0), boundaryL)



"""
VARIATIONAL PROBLEM
"""

# Switching Function Material
def h(_x):
    return _x**3*(Constant(6.0)*_x**2 - Constant(15.0)*_x + Constant(10.0))
def dh(_x):
    return Constant(30.0)*_x*_x*(_x-Constant(1.0))*(_x-Constant(1.0))

# Barrier Function Material
def g(_x):
    return W*_x**2.0*(Constant(1.0) - _x)**2
def dg(_x):
    return W*Constant(2.0)*(_x * (Constant(1.0) - _x) ** 2 - (Constant(1.0) - _x) * _x ** 2)

# Concentration
def cl(_x):
    return exp((_x-el)/A)/(Constant(1.0)+exp((_x-el)/A))
def dcldw(_x):
    return exp((_x+el)/A)/(A*(exp(_x/A)+exp(el/A))**2)
def cs(_x):
    return exp((_x-es)/A)/(Constant(1.0)+exp((_x-es)/A))
def dcsdw(_x):
    return exp((_x+es)/A)/(A*(exp(_x/A)+exp(es/A))**2)

# Susceptibility factor
def chi(_xi,_w):
    return dcldw(_w)*(Constant(1.0)-h(_xi))+dcsdw(_w)*h(_xi)*dv

# Mobility defined by D*c/(R*T), whereR*T is normalized by the chemical potential
# M0*(1-h) is the effective diffusion coefficient; cl*(1-h) is the ion concentration
def D(_xi,_w):
    return M0*(Constant(1.0)-h(_xi))*cl(_w)*(Constant(1.0)-h(_xi))

# Feature of diffusion
def ft(_w):
    return cs(_w)*dv-cl(_w)

# Effective conductivity, Derivative Parsed Material
def Le1(_xi):
    return S1*h(_xi)+S2*(Constant(1.0)-h(_xi))


# Numerical Parameters (time for evolution)
Voltrange=[-0.45]*2750 # (-) plating, (+) stripping.
num_envsteps=len(Voltrange)
t = 0.0
Tfin=0.02 # the time length of each step
Tf = Tfin
dt = 0.02
num_steps = int(Tf/dt) # This will be 1
k = Constant(dt)

# These arrays are now only needed on rank 0 for saving
if rank == 0:
    totaltime=np.zeros((1,num_envsteps))
    totaltime[0,:]=np.arange(0,num_envsteps*Tfin,Tfin)
    Taxis=totaltime[0,:]

# Define variational problem
F1 = (xi-xi_n)/k*v_1*dx + L*kappa*dot(grad(xi),grad(v_1))*dx + L*dg(xi)*v_1*dx + Ls*(exp(phi*AA/Constant(2.0))-Constant(14.89)*cl(w)*(Constant(1.0)-h(xi))*exp(-phi*AA/Constant(2.0)))*dh(xi)*v_1*dx
F2 = chi(xi,w)*(w-w_n)/k*v_2*dx + D(xi, w)*dot(grad(w),grad(v_2))*dx + D(xi, w)*AA*dot(grad(phi),grad(v_2))*dx + (h(xi)-h(xi_n))/k*ft(w)*v_2*dx
F3 = Le1(xi)*dot(grad(phi),grad(v_3))*dx + ft2*(xi-xi_n)/k*v_3*dx
F = F1 + F2 + F3

# Forced parameter
if n_sim == 2:
    Lg = Expression('-0.2*(1.+sin(0.0628*(x[1]-25)))', element = V.ufl_element().sub_elements()[0])
    F = F-Lg*xi.dx(0)*v_1*dx

# Jacobian
J = derivative(F,u)



"""
SOLVE AND SAVE SOLUTIONS
"""

for n in range(num_envsteps):

    phie=Voltrange[n] # the new action (control input).

    # Boundary conditions for phi
    bc_phi1 = DirichletBC(V.sub(2), Constant(phie), boundary0)
    bc_phi2 = DirichletBC(V.sub(2), Constant(0.0), boundaryL)

    # Gather all boundary conditions in a variable
    bcs = [bc_xi1, bc_xi2, bc_c2, bc_phi1, bc_phi2 ] # Dirichlet

    # Time stamp (only on rank 0)
    if rank == 0:
        print("step = ", n, "timestamp =", datetime.fromtimestamp(datetime.timestamp(datetime.now())))

    # Time step
    for n in range(num_steps):

    # Update time
        t+=dt

        # Solve problem
        solve(F == 0, u, bcs, J=J, solver_parameters={"newton_solver":{"absolute_tolerance":1.0e-6, "relative_tolerance":1.0e-6, "maximum_iterations":100}})

        # Update previous solution
        u_n.assign(u)

        if rank == 0:
            set_log_level(LogLevel.INFO)


    # Save u (Parallel-safe HDF5)
    if (n==0 or n==int(round(num_envsteps/4)) or n==int(round(num_envsteps/2)) or n==int(round(num_envsteps*3/4)) or n==int(num_envsteps-1)):
        output_file = HDF5File(mesh.mpi_comm(), f"u{n}.h5", "w")
        output_file.write(u, "solution")
        output_file.close()

    # End message
    if rank == 0:
        print("Completed one step at ", datetime.fromtimestamp(datetime.timestamp(datetime.now())))

# End message
if rank == 0:
    print("Completed at ", datetime.fromtimestamp(datetime.timestamp(datetime.now())))
