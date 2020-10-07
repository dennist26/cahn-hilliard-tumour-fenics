# Cahn-Hilliard model for tumour growth
# by Dennis Trautwein
# ======================
#
# This file is a simplified (!) version of our programs in 2 dimensions.
#
# We modified the demo 'demo_cahn-hilliard.py' from the FEniCS homepage:
'''
 https://fenicsproject.org/docs/dolfin/latest/python/demos.html
'''
#
#
# This demo is implemented in a single Python file,
# which contains both the variational
# forms and the solver.
#
# This example demonstrates the solution of a particular nonlinear
# time-dependent fourth-order equation containing a Cahn--Hilliard system
# modelling tumour growth. In particular it demonstrates the use of
#
# * The built-in Newton solver
# * Advanced use of the base class ``NonlinearProblem``
# * Automatic linearisation
# * A mixed finite element method
# * User-defined Expressions as Python classes
# * Form compiler options
# * Interpolation of functions
#
#
#
# Implementation
# --------------
#
#
# First, the module dolfin is imported::

from dolfin import *
import numpy as np
import random


''' model parameter '''
dt      = 0.001                     # time step size
epsilon = 0.01                    # interface parameter
beta    = 0.1                     # surface tension
lam_P   = 0.1                      # proliferation rate
lam_A   = DOLFIN_EPS                 # apoptosis rate
lam_C   = 1.0                      # consumption rate
eta     = 0.02                    # diffusion/active transport parameter
chi     = 5.0                      # chemotaxis

sig_inf = 1.0                     # Dirichlet boundary value

dynamic = False                     # dynamic or quasi-static equation for sigma

# .. index:: Expression
#
# A class which will be used to represent the initial conditions is then
# created::

''' Class representing the intial conditions '''
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        # if parallelized with MPI
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        # initial value for mu and for sigma
        values[1] = 0.0
        values[2] = 1.0
        # initial value for phi: an ellipse in \R^2
        r = (0.5*x[0]**2 + x[1]**2)**(1./2)
        values[0] = -np.tanh(r/(sqrt(2)*epsilon))
    def value_shape(self):
        # this gives the number of variables we initialize
        return (3,)

# A class which will represent the Cahn-Hilliard in an abstract from for
# use in the Newton solver is now defined. It is a subclass of
# :py:class:`NonlinearProblem <dolfin.cpp.NonlinearProblem>`. ::



''' Class for interfacing with the Newton solver '''
class CahnHilliardTumourEquation(NonlinearProblem):
    def __init__(self, a, L, bc=None):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bc = bc
    def F(self, b, x):
        assemble(self.L, tensor=b)
        if self.bc!=None:       # optional: dirichlet boundary conditions
            self.bc.apply(b,x)
    def J(self, A, x):
        assemble(self.a, tensor=A)
        if self.bc!=None:       # optional: dirichlet boundary conditions
            self.bc.apply(A)

# The constructor (``__init__``) stores references to the bilinear
# (``a``) and linear (``L``) forms. These will used to compute the
# Jacobian matrix and the residual vector, respectively, for use in a
# Newton solver.  The function ``F`` and ``J`` are virtual member
# functions of :py:class:`NonlinearProblem
# <dolfin.cpp.NonlinearProblem>`. The function ``F`` computes the
# residual vector ``b``, and the function ``J`` computes the Jacobian
# matrix ``A``.
#



''' Class where the Dirichlet boundary is specified '''
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# In this example we declare the whole boundary as Dirichlet boundary. One
# could also use a subdomain of the boundary, e.g. with
# ->    return on_boundary and near(x[0],0,DOLFIN_EPS)
# where we only take the part of the boundary with the property ``x_0 = 0``.




#
# .. index::
#    singe: form compiler options; (in Cahn-Hilliard demo)
#
# It is possible to pass arguments that control aspects of the generated
# code to the form compiler. The lines ::


''' Form compiler options '''
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

# tell the form to apply optimization strategies in the code generation
# phase and the use compiler optimization flags when compiling the
# generated C++ code. Using the option ``["optimize"] = True`` will
# generally result in faster code (sometimes orders of magnitude faster
# for certain operations, depending on the equation), but it may take
# considerably longer to generate the code and the generation phase may
# use considerably more memory).
#
# A unit square mesh with 101 (= 100 + 1) vertices in each direction is
# created, and on this mesh a :py:class:`FunctionSpace
# <dolfin.functions.functionspace.FunctionSpace>` ``ME`` is built using
# a pair of linear Lagrangian elements. ::


''' Create mesh and build function space '''
mesh = UnitSquareMesh.create(100, 100, CellType.Type.quadrilateral)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, MixedElement(3*[P1]))



''' Trial and test functions '''
# Trial and test functions of the space ``ME`` are now defined::

# Define trial and test functions
du       = TrialFunction(ME)
q, v, w  = TestFunctions(ME)

# .. index:: split functions
#
# For the test functions, :py:func:`TestFunctions
# <dolfin.functions.function.TestFunctions>` (note the 's' at the end)
# is used to define the scalar test functions ``q``, ``v`` and ``w``. The
# :py:class:`TrialFunction <dolfin.functions.function.TrialFunction>`
# ``du`` has dimension three. Some mixed objects of the
# :py:class:`Function <dolfin.functions.function.Function>` class on
# ``ME`` are defined to represent :math:`u = (c_{n+1}, \mu_{n+1}, sig_{n+1})` and
# :math:`u0 = (c_{n}, \mu_{n}, sig_{n})`, and these are then split into
# sub-functions::

# Define functions
u   = Function(ME)  # current solution
u0  = Function(ME)  # solution from previous converged step

# Split mixed functions
dc, dmu, dsig = split(du)
c,  mu,  sig  = split(u)
c0, mu0, sig0 = split(u0)

# The line ``c, mu, sig = split(u)`` permits direct access to the components
# of a mixed function. Note that ``c``, ``mu`` and ``sig`` are references for
# components of ``u``, and not copies.
#
# .. index::
#    single: interpolating functions; (in Cahn-Hilliard demo)
#
# Initial conditions are created by using the class defined at the
# beginning of the demo and then interpolating the initial conditions
# into a finite element space::


''' initial conditions '''
# Create intial conditions and interpolate:
u_init = InitialConditions(degree=1)
u.interpolate(u_init)
u0.interpolate(u_init)

# The first line creates an object of type ``InitialConditions``.  The
# following two lines make ``u`` and ``u0`` interpolants of ``u_init``
# (since ``u`` and ``u0`` are finite element functions, they may not be
# able to represent a given function exactly, but the function can be
# approximated by interpolating it in a finite element space).


''' Define the nonlinear functions '''
# now we define the nonlinear functions:
# DPsi -> derivative of the double well potential
# m_phi -> mobility function
# h_phi -> interpolation function

DPsi = c**3 - c
m_phi = 0.5*(1+c0)**2 + 5e-06
h_phi = 0.5*(1+c0)


''' Weak statement of the equations '''
# now we give the weak formulation of the time discretized system of equations:
''' first equation, testing with q '''
L01 = c*q*dx - c0*q*dx
L02 = dt*m_phi*dot(grad(mu), grad(q))*dx
L03 = - dt*(lam_P*sig-lam_A)*h_phi*q*dx
L0 = L01 + L02 + L03

''' second equation, testing with v '''
L11 = mu*v*dx - beta/epsilon*DPsi*v*dx
L12 = - beta*epsilon*dot(grad(c), grad(v))*dx
L13 = chi*sig*v*dx
L1 = L11 + L12 + L13

''' third equation, testing with w '''
L21 = dot(grad(sig), grad(w))*dx
if dynamic:    # dynamic or quasi-static
    L21  += 1./dt*(sig*w*dx - sig0*w*dx)
L22 = - eta*dot(grad(c), grad(w))*dx
L23 = lam_C*sig*h_phi*w*dx
L2 = L21 + L22 + L23

# This is a statement of the time-discrete equations presented as part
# of the problem statement, using UFL syntax. The linear forms for the
# two equations can be summed into one form ``L``, and then the
# directional derivative of ``L`` can be computed to form the bilinear
# form which represents the Jacobian matrix::

# Compute directional derivative about u in the direction of du (Jacobian)
L = L0 + L1 + L2
a = derivative(L, u, du)


''' Dirichlet boundary conditions for sigma '''
bc = DirichletBC(ME.sub(2), sig_inf, DirichletBoundary())



# .. index::
#    single: Newton solver; (in Cahn-Hilliard demo)
#
# The DOLFIN Newton solver requires a :py:class:`NonlinearProblem
# <dolfin.cpp.NonlinearProblem>` object to solve a system of nonlinear
# equations. Here, we are using the class ``CahnHilliardTumourEquation``,
# which was declared at the beginning of the file, and which is a
# sub-class of :py:class:`NonlinearProblem
# <dolfin.cpp.NonlinearProblem>`. We need to instantiate objects of both
# ``CahnHilliardTumourEquation`` and :py:class:`NewtonSolver
# <dolfin.cpp.NewtonSolver>`::

''' Create nonlinear problem and Newton solver '''
problem = CahnHilliardTumourEquation(a, L, bc)
solver = NewtonSolver()
solver.parameters["linear_solver"] = "lu"
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# The string ``"lu"`` passed to the Newton solver indicated that an LU
# solver should be used.  The setting of
# ``parameters["convergence_criterion"] = "incremental"`` specifies that
# the Newton solver should compute a norm of the solution increment to
# check for convergence (the other possibility is to use ``"residual"``,
# or to provide a user-defined check). The tolerance for convergence is
# specified by ``parameters["relative_tolerance"] = 1e-6``.
#
# To run the solver and save the output to a VTK file for later visualization,
# the solver is advanced in time from :math:`t_{n}` to :math:`t_{n+1}` until
# a terminal time :math:`T` is reached::

# Output file
file = File("output.pvd", "compressed")

# Step in time
N = 50
t = 0.0
T = N*dt
j = 0
while (j < N):
    j += 1
    t += dt
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())
    file << (u.split()[0], t)
    print("\nCompleted iteration number:  ", j, " of ", N, "\n")


# The string ``"compressed"`` indicates that the output data should be
# compressed to reduce the file size. Within the time stepping loop, the
# solution vector associated with ``u`` is copied to ``u0`` at the
# beginning of each time step, and the nonlinear problem is solved by
# calling :py:func:`solver.solve(problem, u.vector())
# <dolfin.cpp.NewtonSolver.solve>`, with the new solution vector
# returned in :py:func:`u.vector() <dolfin.cpp.Function.vector>`. The
# ``c`` component of the solution (the first component of ``u``) is then
# written to file at every time step.
