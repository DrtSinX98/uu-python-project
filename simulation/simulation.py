"""
FEniCS program: Phase-field model of lithium dendrite growth.

  Based in the article of Hong and Viswanathan:
  https://doi.org/10.1021/acsenergylett.8b01009

------------------------------------------------------------------
  rbkrgb                                                   2020
------------------------------------------------------------------
"""

from fenics import *
from datetime import datetime


# Start message
print("Starting 01_dend_simulation.py at ", datetime.fromtimestamp(datetime.timestamp(datetime.now())))


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

# Applied voltage
phie = -0.45

# Normalized Parameters 
L = Constant(6.25)
kappa = Constant(0.3)
Ls = Constant(0.001)
alpha = Constant(0.5)
AA = Constant(38.69)   # nF/RT
W = Constant(2.4)
es = Constant(-13.8)
el = Constant(2.631)
A = Constant(1.0)      # R*T/R*T
c0 = Constant(1.0/14.89)
dv = Constant(5.5)     # Csm/Clm
M0 = Constant(317.9)
S1 = Constant(1000000)
S2 = Constant(1.19)
ft2 = Constant(0.0074)



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

# Boundary conditions for phi
bc_phi1 = DirichletBC(V.sub(2), Constant(phie), boundary0)
bc_phi2 = DirichletBC(V.sub(2), Constant(0.0), boundaryL)

# Gather all boundary conditions in a variable
bcs = [bc_xi1, bc_xi2, bc_c2, bc_phi1, bc_phi2 ] # Dirichlet



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
t = 0.0
Tf = 108.0
dt = 0.02
num_steps = int(Tf/dt)
k = Constant(dt)


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

# Create progress bar
progress = Progress('Time-stepping', num_steps)

# Time step
for n in range(num_steps):

    # Update time
    t+=dt

    # Solve problem
    solve(F == 0, u, bcs, J=J, solver_parameters={"newton_solver":{"absolute_tolerance":1.0e-6, "relative_tolerance":1.0e-6, "maximum_iterations":100}})

    # Update previous solution
    u_n.assign(u)

    # Timestamp
    print("step = ", n, "timestamp =", datetime.fromtimestamp(datetime.timestamp(datetime.now())))

    # Save solution each 5 seconds of simulation
    if (t % 5.0 < 0.001 or t % 5.0 > 4.999):
        fsolution=f"saved/sim{n_sim+1}/u_t{round(float(t))}.xml"
        File(fsolution) << u

        # Timestamp
        print("Solution ", fsolution, " saved at ", datetime.fromtimestamp(datetime.timestamp(datetime.now())))

    # Update progress bar
    set_log_level(LogLevel.PROGRESS)
    progress+=1
    set_log_level(LogLevel.ERROR)

# Save last solution
fsolution=f"saved/sim{n_sim+1}/u_t{round(float(t))}_final.xml"
File(fsolution) << u


# End message
print("Completed at ", datetime.fromtimestamp(datetime.timestamp(datetime.now())))
