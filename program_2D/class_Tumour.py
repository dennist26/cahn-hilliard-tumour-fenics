from dolfin import *
from settings import *
from class_initialconditions import *


class Tumour:
    ''' model parameters '''
    num_eq  = 3                       # number of equations
    epsilon = 0.01                    # interface parameter, epsilon > dt (!)
    beta    = 0.1                     # surface tension
    lam_P   = 0.1                      # proliferation rate
    lam_A   = DOLFIN_EPS                 # apoptosis rate
    lam_C   = 1.0                      # consumption rate
    eta     = 0.02                    # diffusion parameter
    chi     = 5.0                      # chemotaxis

    sig_B   = 1.0                     # dirichlet boundary value for sigma
    sig_inf = 1.0                       # Robin boundary conditions
    K       = 1000.0

    sig0    = 1.0                     # initial condition for sigma
    dynamic = True                     # dynamic or quasi-static eq for sigma



    boundary_conditions = {1: {'Dirichlet': sig_B},             #  sig = sig_B
                           2: {'Robin':     (K, sig_inf)},      #  del_n sig = K(sig_inf - sig)
                           0: {'Neumann':    0} }               #  del_n sig = 0


    ''' initialize problem '''
    def init_problem(self, dt, u, u0):
        ''' variables '''
        ME              = u.function_space()
        mesh            = u.function_space().mesh()
        du              = TrialFunction(ME)
        q, v, w         = TestFunctions(ME)
        dc, dmu, dsig   = split(du)
        c,  mu, sig     = split(u)
        c0, mu0, sig0   = split(u0)

        ''' nonlinear functions '''
        c    = variable(c)
        #f    = 0.25*(c+1)**2*(1-c)**2
        #DPsi = diff(f, c)
        DPsi = (c**3-c0)        # derivative of potential, dpsi/dphi
        m_phi = 0.5*(1+c0)**2 + 5e-06  # mobility function m(phi)
        h_phi = 0.5*(1+c0)       # interpolation function h(phi)
        ''' first equation, testing with q '''
        L01 = c*q*dx - c0*q*dx
        L02 = dt*m_phi*dot(grad(mu), grad(q))*dx
        L03 = - dt*(self.lam_P*sig-self.lam_A)*h_phi*q*dx
        L0 = L01 + L02 + L03
        ''' second equation, testing with v '''
        L11 = mu*v*dx - self.beta/self.epsilon*DPsi*v*dx
        L12 = - self.beta*self.epsilon*dot(grad(c), grad(v))*dx
        L13 = self.chi*sig*v*dx
        L1 = L11 + L12 + L13
        ''' third equation, testing with w '''
        L21 = dot(grad(sig), grad(w))*dx
        if self.dynamic:    # dynamic or quasi-static
            L21  += 1./dt*(sig*w*dx - sig0*w*dx)
        L22 = - self.eta*dot(grad(c), grad(w))*dx
        L23 = self.lam_C*sig*h_phi*w*dx
        L2 = L21 + L22 + L23

        ''' set boundary conditions for sigma '''
        # Mark boundaries
        boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 9999)
        boundary_markers.set_all(9999)
        bx0 = BoundaryX0()
        bx1 = BoundaryX1()
        bx0.mark(boundary_markers, config.boundary_x0)
        bx1.mark(boundary_markers, config.boundary_x1)

        dirichlet = False     # sonst ständig Warnung falls keine Dirichlet BC
        dirichlet = dirichlet or (config.boundary_x0==1) or  (config.boundary_x1==1)

        if discr.dimension>1:
            by0 = BoundaryY0()
            by1 = BoundaryY1()
            by0.mark(boundary_markers, config.boundary_y0)
            by1.mark(boundary_markers, config.boundary_y1)
            dirichlet = dirichlet or (config.boundary_y0==1) or  (config.boundary_y1==1)
        if discr.dimension>2:
            bz0 = BoundaryZ0()
            bz1 = BoundaryZ1()
            bz0.mark(boundary_markers, config.boundary_z0)
            bz1.mark(boundary_markers, config.boundary_z1)
            dirichlet = dirichlet or (config.boundary_z0==1) or  (config.boundary_z1==1)

        # Redefine boundary integration measure
        ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

        # Collect Neumann integrals
        for i in self.boundary_conditions:
            if 'Neumann' in self.boundary_conditions[i]:
                if self.boundary_conditions[i]['Neumann'] != 0:
                    g = self.boundary_conditions[i]['Neumann']
                    L2 += g*w*ds(i)

        # Collect Robin integrals
        for i in self.boundary_conditions:
            if 'Robin' in self.boundary_conditions[i]:
                K, sig_inf = self.boundary_conditions[i]['Robin']
                L2 += K*(sig-sig_inf)*w*ds(i)

        # Collect Dirichlet conditions
        if dirichlet:
            bc = DirichletBC(ME.sub(2), self.sig_B, DirichletBoundary())


        ''' Compute directional derivative about u in the direction of du (Jacobian) '''
        L = L0 + L1 + L2
        a = derivative(L, u, du)
        ''' Create nonlinear problem and Newton solver '''
        if dirichlet:
            problem = MyNonlinearProblem(a,L,bc)
        else:
            problem = MyNonlinearProblem(a,L)

        return problem

model = Tumour()


''' class for initial conditions '''
class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def value_shape(self):
        return (model.num_eq,)
    def eval(self, values, x):
        values[0] = function_5(x, model)
        if model.num_eq >=2:
            values[1] = 0.0
        if model.num_eq >= 3:
            values[2] = model.sig0
