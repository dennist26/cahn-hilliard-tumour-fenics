from dolfin import *

class Discretization_Parameter:
    ''' mesh parameters '''
    dimension           = 3             # dimension (1D, 2D or 3D)
    length              = 3           # length of mesh
    N_mesh              = 64            # number gridpoints for mesh
    mesh_sphere         = False          # cube or sphere
    ''' FE function parameter '''
    quadr_degree        = 4              # degree of quadrature
    quadr_lump          = True          # Mass lumping (only for polynom_degree=1)
    polynom_degree      = 1             # degree of polynoms
    ''' time parameters '''
    dt                  = 1.0e-03            # time step
    fix_time_interval   = False                     # use Tmax if True, or N_time if False
    N_time              = 10000                         # number of total time iterations
    Tmax                = 5                       # final time
    if fix_time_interval:
        N_time      = int(Tmax/dt)
    else:
        Tmax        = N_time*dt
    ''' parameters for mesh adaption '''
    adapt               = True
    adapt_delta         = 0.1                   # refine where |phi|< 1-adapt_delta
    adapt_freq          = 10                     # coarse mesh every adapt_freq steps
    adapt_N_coarse      = 10                    # coarsest mesh
    adapt_N_fine        = 160                   # finest mesh
    adapt_it_max        = 5                    # such that there is no infinity loop

discr = Discretization_Parameter()


class Config_Parameter:
    ''' init from file '''
    init_from_file       = False                # -> mesh.xml & u.xml
    init_degree          = 3
    init_path            = "data/konvergenz_stat_K=1000/init"
    ''' save output '''
    save_mesh_to_png     = False
    save_as_hdf5         = False          # this for calculating errors
    save_as_xml          = False         # this for saving initial conditions
    save_as_vtu          = True         # this to visualize with paraview
    #output_path          = "/home/fenics/data/Tumour_4/data" + str(discr.dimension) + "D__N=" + str(discr.N_mesh)
    output_path          = "data/test_3D_2/data" + str(discr.dimension) + "D__N=" + str(discr.N_mesh)

    output_freq          = 50            # iteration frequency with that the output is saved
    ''' plot during run '''
    # Note: Plotting during runtime is maybe not supported any more, depending on
    # the versions of FEniCS and Matplotlib.
    plot_init            = False          # plot initial conditions?
    plot_solution        = False         # plot some solutions in real time?
    plot_freq            = 50            # iteration frequency with that the current solution is plotted
    ''' Dirichlet boundary - specify on which side dirichlet bc are fulfilled '''
    boundary_x0         = 0
    boundary_x1         = 1
    boundary_y0         = 0
    boundary_y1         = 1
    boundary_z0         = 0
    boundary_z1         = 1
    #boundary_conditions = {1: {'Dirichlet': sig_B},             #  sig = sig_B
    #                       2: {'Robin':     (K, sig_inf)},      #  del_n sig = K(sig - sig_inf)
    #                       0: {'Neumann':    0} }               #  del_n sig = 0

config = Config_Parameter()



'''
######################################################################
'''

class MyNonlinearProblem(NonlinearProblem):
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

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        ''' spherical mesh '''
        if discr.mesh_sphere:
            return on_boundary

        ''' interval/rectangle/cubic mesh '''
        dim = discr.dimension
        bool = False
        tol = 1e-14
        # 1D
        if config.boundary_x0==1:
            bool = bool or near(x[0],0.0,tol)
        if config.boundary_x1==1:
            bool = bool or near(x[0],discr.length,tol)
        # 2D
        if dim>=2 and config.boundary_y1==1:
            bool = bool or near(x[1],discr.length,tol)
        if dim>=2 and config.boundary_y0==1:
            bool = bool or near(x[1],0.0,tol)
        # 3D
        if dim==3 and config.boundary_z1==1:
            bool = bool or near(x[2],discr.length,tol)
        if dim==3 and config.boundary_z0==1:
            bool = bool or near(x[2],0.0,tol)
        return (bool and on_boundary)



class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], discr.length, DOLFIN_EPS)

class BoundaryY0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, DOLFIN_EPS)

class BoundaryY1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], discr.length, DOLFIN_EPS)

class BoundaryZ0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], 0, DOLFIN_EPS)

class BoundaryZ1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[2], discr.length, DOLFIN_EPS)
