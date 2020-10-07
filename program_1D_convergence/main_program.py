from dolfin import *
import matplotlib.pyplot as plt
from os import path
from shutil import copyfile

# import necessary files
from settings import *
from useful_functions import *
from class_Tumour import *


''' initialize form compiler '''
parameter_form_compiler(discr)

# for mpi
mpi_comm = MPI.comm_world
my_rank = MPI.rank(mpi_comm)


''' print some info '''
print_info(discr, config, model, my_rank)


''' initial conditions '''
if config.init_from_file:
    # init from file
    mesh_init = Mesh(config.init_path + "/mesh.xml")
    P1_init = FiniteElement("Lagrange", mesh_init.ufl_cell(), config.init_degree)
    ME_init = FunctionSpace(mesh_init,MixedElement(model.num_eq*[P1_init]))
    u_init = Function(ME_init, config.init_path+"/u.xml")
else:
    u_init = InitialConditions(degree=discr.polynom_degree)

''' Create mesh, optional: create adapted mesh '''
if discr.adapt:
    mesh = refine_mesh(discr, model, mpi_comm, u_init, 0)
else:
    mesh = init_mesh(discr, mpi_comm)

''' Initialize functions '''
u, u0 = init_function(discr, config, model, mesh)
u.interpolate(u_init)

''' initialize problem '''
problem = model.init_problem(discr.dt, u, u0)
solver = init_Newton_solver()

j = 0
t = 0
''' save initial conditions into output files '''
file0, file1, file2 = open_vtu(config, model)
save_output_vtu(discr,config,model,u,j,t,file0,file1,file2)
save_output(discr, config, model, u, mesh, j, t)

''' time iteration '''
while (j < discr.N_time):
    j += 1
    t += discr.dt

    ''' solve '''
    u0.vector()[:] = u.vector()
    solver.solve(problem, u.vector())

    if my_rank==0:
        print("\nCompleted iteration number:  ", j, " of ", discr.N_time, "\n")

    ''' optional: save output '''
    save_output(discr, config, model, u, mesh, j, t)
    save_output_vtu(discr,config,model,u,j,t,file0,file1,file2)

    ''' optional: adapt mesh '''
    if discr.adapt:
        mesh = refine_mesh(discr, model, mpi_comm, u, j)
        u, u0 = init_function(discr, config, model, mesh, u, u0)
        problem = model.init_problem(discr.dt, u, u0)
        solver = init_Newton_solver()
