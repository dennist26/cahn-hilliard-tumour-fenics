from dolfin import *
import time
from mshr import *      # more complex meshes
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


''' Calculate errors '''
def my_errornorm(uref, u, norm_type='L2', degree_rise=3):
    '''
    Info:
    uref must have a finer mesh or a higher polynomial degree than u.
    Otherwise too low errors (too optimistic convergence rates) will occur.
    E.g. of uref and u are linear FE functions on the same mesh, then you will
    probably get a convergence rate of 2 instead of 1 for the H^1_0 error.
    '''

    if norm_type!='Linf' and degree_rise>0:
        value = errornorm(uref, u, norm_type, degree_rise)
        return value


    ''' get finer mesh - uref can be a FE function or an expression '''
    if isinstance(uref, dolfin.function.function.Function):
        mesh = uref.function_space().mesh()
    if isinstance(uref, dolfin.function.expression.Expression):
        mesh = u.function_space().mesh()
        mesh = refine(mesh)

    ''' interpolate u on the finer mesh '''
    degree = u.function_space().ufl_element().degree()
    W = FunctionSpace(mesh, 'P', degree)
    u_W = interpolate(u, W)
    uref_W = interpolate(uref, W)

    ''' if degree_rise>0 we will use the built in function '''

    ''' error function '''
    e_W = Function(W)
    e_W.vector()[:] = np.array(uref_W.vector()) - np.array(u_W.vector())

    ''' norm of error function '''
    if norm_type == 'H10':
        error = dot(grad(e_W), grad(e_W))*dx
        value = sqrt(abs(assemble(error)))
    elif norm_type == 'H1':
        error = dot(grad(e_W), grad(e_W))*dx + e_W**2*dx
        value = sqrt(abs(assemble(error)))
    elif norm_type == 'Linf':
        value = abs(np.array(e_W.vector())).max()
    else:   #L2
        error = e_W**2*dx
        value = sqrt(abs(assemble(error)))
    return value



''' Compute convergence rate '''
def convergence_rate(N_vec,L2_vec,H10_vec,Linf_vec,path):
    file = open(path+"convergence_rate.txt","w")
    print('\nConvergence rates:  N   L2Linf   L2L2   L2H10')
    for i in range(len(L2_vec)-1):
        r_L2 = ln(L2_vec[i+1]/L2_vec[i]) / ln(N_vec[i]/N_vec[i+1])
        r_H10 = ln(H10_vec[i+1]/H10_vec[i]) / ln(N_vec[i]/N_vec[i+1])
        r_Linf = ln(Linf_vec[i+1]/Linf_vec[i]) / ln(N_vec[i]/N_vec[i+1])

        print("N="+str(N_vec[i])+"    " + str(r_Linf) + "   " + str(r_L2) +
            "   " + str(r_H10))


        file.write(str(N_vec[i]) + "  " + str(r_Linf) + "  "
            + str(r_L2) + "  " + str(r_H10) + "\n")

    file.close()

    return



''' initialze form compiler '''
def parameter_form_compiler(discr):
    # Form compiler options
    parameters["ghost_mode"] = "shared_facet"
    parameters["form_compiler"]["optimize"]     = True
    parameters["form_compiler"]["cpp_optimize"] = True

    parameters['allow_extrapolation'] = False        # test
    parameters["std_out_all_processes"] = False     # turn off redundant output in parallel


    if discr.quadr_lump:
        ''' Mass lumping '''
        '''
        Maybe this won't be supported in future...
        For another way see: https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/mass_lumping.html
        or "mass_lumping_test.py".
        '''
        import warnings
        from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
        warnings.simplefilter("once", QuadratureRepresentationDeprecationWarning)
        parameters["form_compiler"]["representation"] = "quadrature"
        parameters["form_compiler"]["quadrature_rule"] = "vertex"
        parameters["form_compiler"]["quadrature_degree"] = 1    # degree of quadrature

    else:
        ''' Gaussian quadrature rules '''
        parameters["form_compiler"]["representation"] = "uflacs"   # this helps that "large" problems compile
        parameters["form_compiler"]["quadrature_degree"] = discr.quadr_degree    # degree of quadrature



''' print some stuff '''
def print_info(discr, config, model, mpi_rank):
    if mpi_rank>0:
        return
    print('\n')
    print('Model:                ', model.__class__.__name__)
    print('\nDimension:            ', discr.dimension)
    if discr.mesh_sphere:
        print('Spherical mesh:       ', discr.mesh_sphere)
    if discr.adapt:
        print('Mesh refinement:      ', discr.adapt)
        print('Coarsest N_mesh:      ', discr.adapt_N_coarse)
        print('Finest N_mesh:        ', discr.adapt_N_fine)
    else:
        print('N_mesh:               ', discr.N_mesh)
    print('\nPolynom degree:       ', discr.polynom_degree)
    print('Quadratur degree:     ', discr.quadr_degree)
    if discr.quadr_lump:
        print('Mass lumping:         ', discr.quadr_lump)
    print('\ndt:                   ', discr.dt)
    print('N_time:               ', discr.N_time)
    print('Tmax:                 ', discr.Tmax)
    print('\nSave output:          ', config.save_as_vtu or config.save_as_xml or config.save_as_hdf5)
    print('Init from file:       ', config.init_from_file)
    print('\n')
    time.sleep(0)



'''
mesh adaption algorithm
'''
def refine_mesh(discr, model, mpi_comm, u_old, j):

    '''
    Due to complexity of how a mesh is stored, local mesh coarsening is
    not supported in fenics. Instead we create a new coarse mesh after
    discr.adapt_freq steps, and refine it locally.
    '''

    ''' check if u_old is a function '''
    if isinstance(u_old, dolfin.function.function.Function):
        init = False
    else:
        init=True

    ''' refine old mesh, or create new mesh '''
    if (j+discr.adapt_freq-1)%discr.adapt_freq>0 and not init:
        mesh_tmp = u_old.function_space().mesh()
    else:
        mesh_tmp = init_mesh(discr, mpi_comm, discr.adapt_N_coarse)

    ''' temporal function '''
    V_tmp = FunctionSpace(mesh_tmp,"Lagrange", discr.polynom_degree)
    u_tmp = Function(V_tmp)
    if init:
        P1_tmp = FiniteElement("Lagrange", mesh_tmp.ufl_cell(), discr.polynom_degree)
        ME_tmp = FunctionSpace(mesh_tmp,MixedElement(model.num_eq*[P1_tmp]))
        u_init_tmp = Function(ME_tmp)
        u_init_tmp.interpolate(u_old)
        u_tmp.interpolate(u_init_tmp.split()[0])
    else:
        u_tmp.interpolate(u_old.split()[0])

    hmin = discr.length / discr.adapt_N_fine *1.000001 #* sqrt(discr.dimension)
    ''' refine mesh '''
    i = 0
    i_max = discr.adapt_N_fine / discr.adapt_N_coarse + discr.adapt_it_max
    bool_tmp = True
    while bool_tmp and i<i_max:
        i += 1
        # refine
        bool_break = True
        markers = MeshFunction("bool",mesh_tmp,discr.dimension-1,False)

        if discr.dimension==1:  # in 1 dimension, it works only like this...
            for c in cells(mesh_tmp):
                p = c.midpoint()
                h = c.h()
                #h = min([edge.length() for edge in edges(c)])
                if  u_tmp(p)<1-discr.adapt_delta  and u_tmp(p)>-1+discr.adapt_delta  and h>=hmin-DOLFIN_EPS:
                    markers[c] = True
                    bool_break = False
        else:
            for facet in facets(mesh_tmp):   # facets(mesh), not cells(mesh) !!!
                p = facet.midpoint()
                h = min([edge.length() for edge in edges(facet)])
                if  u_tmp(p)<1-discr.adapt_delta  and u_tmp(p)>-1+discr.adapt_delta  and h>=hmin-DOLFIN_EPS:
                    markers[facet] = True
                    bool_break = False

        bool_tmp = not bool_break
        mesh_tmp = refine(mesh_tmp,markers)
        print("mesh: ", mesh_tmp.hmax(), mesh_tmp.hmin())
        # interpolate fine solution on refined coarse mesh
        V_tmp = FunctionSpace(mesh_tmp,"Lagrange", discr.polynom_degree)
        u_tmp = Function(V_tmp)
        if init:
            P1_tmp = FiniteElement("Lagrange", mesh_tmp.ufl_cell(), discr.polynom_degree)
            ME_tmp = FunctionSpace(mesh_tmp,MixedElement(model.num_eq*[P1_tmp]))
            u_init_tmp = Function(ME_tmp)
            u_init_tmp.interpolate(u_old)
            u_tmp.interpolate(u_init_tmp.split()[0])
        else:
            u_tmp.interpolate(u_old.split()[0])

    return mesh_tmp


''' initialize function spaces and functions '''
def init_function(discr, config, model, mesh, u_old=None, u0_old=None):
    ''' initialize function '''
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), discr.polynom_degree)
    ME = FunctionSpace(mesh,MixedElement(model.num_eq*[P1]))
    u = Function(ME)
    u0 = Function(ME)

    if u_old!=None:
        u.interpolate(u_old)
    if u0_old!=None:
        u0.interpolate(u0_old)

    return (u, u0)


''' initialize mesh '''
def init_mesh(discr, mpi_comm, N=None):
    ''' initialize mesh '''
    if N==None:
        N  = discr.N_mesh

    length = discr.length
    dim    = discr.dimension
    if dim==3:
        if discr.mesh_sphere:
            domain = Sphere(Point(0, 0, 0), Length)
            mesh = generate_mesh(domain, N)
        else:
            mesh = BoxMesh(mpi_comm, Point(0.0, 0.0, 0.0), Point(length, length, length), N,N,N)
    elif dim==2:
        if discr.mesh_sphere:
            domain = Circle(Point(0, 0), length)
            mesh = generate_mesh(domain, N)
        else:
            mesh = RectangleMesh.create(mpi_comm,[Point(0,0), Point(length,length)], [N,N], CellType.Type.triangle, "crossed")
    else:
        mesh = IntervalMesh(mpi_comm, N, 0, length)
    return mesh


''' configurate the properties of the Newton Solver '''
def init_Newton_solver():
    solver = NewtonSolver()
    #solver.parameters['convergence_criterion'] = 'incremental'
    solver.parameters['relative_tolerance'] = 1e-6
    solver.parameters['linear_solver'] =  'lu'
    # best: lu with default, gmres with ilu, mumps with default, bicgstab with default
    ''' possible linear solvers:
    print(list_linear_solver_methods())
    # 'bicgstab'        # Biconjugate gradient stabilized method, works for N>400 #with default
    # 'cg'              # conjugate gradient method
    # 'default'         # default
    # 'gmres'           # Generalized minimal residual method       #with ilu
    # 'lu'              # direct sparse LU solver
    # 'minres'          # Minimal residual method
    # 'mumps'           # MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)  #like lu
    # 'petsc'           # PETSc built in LU solver (=lu)
    # 'richardson'      # Richardson method
    # 'superlu'         # superlu
    # 'tfqmr'           # Transpose-free quasi-minimal residual method
    # 'umfpack'         # UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)
    '''
    solver.parameters['preconditioner'] = 'default'
    ''' possible preconditioners (only iterative solvers)
    print(list_krylov_solver_preconditioners())
    # 'amg'               # Algebraic multigrid
    # 'default'           # default
    # 'hypre_amg'         # Hypre algebraic multigrid (BoomerAMG)
    # 'hypre_euclid'      # Hypre parallel incomplete LU factorization
    # 'hypre_parasails'   # Hypre parallel sparse approximate inverse
    # 'icc'               # Incomplete Cholesky factorization
    # 'ilu'               # Incomplete LU factorization
    # 'jacobi'            # Jacobi iteration
    # 'none'              # No preconditioner
    # 'petsc_amg'         # PETSc algebraic multigrid
    # 'sor'               # Successive over-relaxation
    '''
    solver.parameters['krylov_solver']['absolute_tolerance'] = 1E-7
    solver.parameters['krylov_solver']['relative_tolerance'] = 1E-4
    solver.parameters['krylov_solver']['maximum_iterations'] = 1000

    return solver


''' save the output '''
def save_output(discr, config, model, u, mesh, j, t):
    ''' save mesh to png '''
    if config.save_mesh_to_png and (j%config.output_freq==0 or j==discr.N_time):
        plt.figure()
        plot(mesh)
        plt.savefig(config.output_path+"mesh__"+str(j)+".png")
        # for vector graphic: *.eps or *.svg (high resolution)
        # fig.savefig('myimage.svg', format='svg', dpi=1200)

    ''' save mesh '''
    if j==0:
        if config.save_as_hdf5:
            file_mesh_hdf5 =  HDF5File(mesh.mpi_comm(), config.output_path + "/mesh.h5","w")
            file_mesh_hdf5.write(mesh, "/mesh")
            file_mesh_hdf5.close()
        if config.save_as_xml:
            file_mesh_xml = File(config.output_path + "/mesh.xml")
            file_mesh_xml << mesh

    ''' save functions '''
    if config.save_as_hdf5 and (j%config.output_freq==0 or j==discr.N_time):
        file_u_hdf5 = HDF5File(mesh.mpi_comm(),config.output_path + "/u_" + str(j) + ".h5","w")

        V = FunctionSpace(mesh,"Lagrange", 1)
        Z = Function(V)
        Z.interpolate(u.split()[0])
        file_u_hdf5.write(Z, "/phi")
        if model.num_eq>=2:
            Z.interpolate(u.split()[1])
            file_u_hdf5.write(Z, "/mu")
        if model.num_eq>=3:
            Z.interpolate(u.split()[2])
            file_u_hdf5.write(Z, "/sigma")
        file_u_hdf5.close()

    if config.save_as_xml and (j%config.output_freq==0 or j==discr.N_time):
        file_u_xml = File(config.output_path + "/u_" + str(j) + ".xml")
        file_u_xml << u
        '''
        V = FunctionSpace(mesh,"Lagrange", 1)
        Z = Function(V)
        Z.interpolate(u.split()[0])
        file_phi_xml = File(config.output_path + "/phi_" + str(j) + ".xml")
        file_phi_xml << Z
        if model.num_eq>=2:
            Z.interpolate(u.split()[1])
            file_mu_xml = File(config.output_path + "/mu_" + str(j) + ".xml")
            file_mu_xml << Z
        if model.num_eq>=3:
            Z.interpolate(u.split()[2])
            file_sigma_xml = File(config.output_path + "/sigma_" + str(j) + ".xml")
            file_sigma_xml << Z
        '''


''' open a vtu-file with FEniCS ''' # fenics closes it automatically when finished
def open_vtu(config, model):
    file0 = None
    file1 = None
    file2 = None
    if not config.save_as_vtu:
        return (file0, file1, file2)
    file0 = File(config.output_path + "/phi.pvd", "compressed")
    if model.num_eq >= 2:
        file1 = File(config.output_path + "/mu.pvd", "compressed")
    if model.num_eq >=3:
        file2 = File(config.output_path + "/sigma.pvd", "compressed")
    return (file0, file1, file2)

''' save output in a vtu-file with FEniCS '''
def save_output_vtu(discr, config, model, u, j, t, file0, file1=None, file2=None):
    if config.save_as_vtu and (j%config.output_freq==0 or j==discr.N_time):
        file0 << (u.split()[0], t)
        if file1!=None and model.num_eq>=2:
            file1 << (u.split()[1], t)
        if file2!=None and model.num_eq>=3:
            file2 << (u.split()[2], t)
