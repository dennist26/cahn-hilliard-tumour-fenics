from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from useful_functions import *
from settings import discr
parameter_form_compiler(discr)


mpi_comm = MPI.comm_world

N_vec = [32, 64, 128, 256]
N_ref = 1024
#dt_ref = 16      # von N_ref ist nur jedes dt_ref'te gespeichert

numN = len(N_vec)
T = 0.1
mu = 1.0            # the ratio between h^dt_potenz and dt
dt_potenz = 2        # dt = mu*h**dt_potenz

deg_rise = 1        #degree rise for fenics
degree_uref = 3     #degree of reference solution

var_vec = ["phi","mu","sigma"]#, "sigma"]
path = "data/"




L2_vec = [0]*(len(N_vec))
H10_vec = [0]*(len(N_vec))
Linf_vec = [0]*(len(N_vec))


for var in var_vec:
    name_out = var + "_error"
    var_name = "/" + var
    #########################################################
    print("\n######## ||"+var+"_k-"+var+"_*|| ############")
    file = open(path + name_out + "_ref.txt","w")

    error_L2L2 = 0
    error_L2H10 = 0
    error_LinfL2 = 0

    # lade mesh von u_*
    N1 = N_ref
    mesh1 = Mesh()
    hdf5 = HDF5File(mpi_comm, path+"data1D__N="+str(N1)+"/mesh.h5", 'r')
    hdf5.read(mesh1, '/mesh', False)
    hdf5.close()
    V1 = FunctionSpace(mesh1, "Lagrange", degree_uref)
    u1 = Function(V1)

    # Iteration über die N bzw. h=1/N
    for i in range(0, numN):
        # lade mesh von u_i
        N0 = N_vec[i]
        mesh0 = Mesh()
        hdf5 = HDF5File(mpi_comm, path+"data1D__N="+str(N0)+"/mesh.h5", 'r')
        hdf5.read(mesh0, '/mesh', False)
        hdf5.close()
        V0 = FunctionSpace(mesh0, "Lagrange", 1)
        u0 = Function(V0)

        # dt entspricht dem dt der Lösung u_i
        dt = mu/(N0**dt_potenz)
        numT = int(T*N0**dt_potenz/mu)

        error_L2L2 = 0
        error_L2H10 = 0
        error_L2H1 = 0
        error_LinfL2 = 0

        ########################## zeitliche Iteration
        for j in range(0,numT):
            # lade u_i(j*dt_0)
            hdf5 = HDF5File(mpi_comm, path + "data1D__N="+str(N0)+"/u_" + str(j) +".h5", 'r')
            hdf5.read(u0, var_name)
            hdf5.close()
            # lade u_{i+1}(j*dt_0)=u_{i+1}(j*4*dt_1)
            index = int(j*((N1/N0)**dt_potenz))
            hdf5 = HDF5File(mpi_comm, path + "data1D__N="+str(N1)+"/u_" + str(index) +".h5", 'r')
            #hdf5 = HDF5File(mpi_comm, path + "data1D__N="+str(N1)+"/u_" + str(dt_ref*j*(2**dt_potenz)**(numN-i-1)) +".h5", 'r')
            hdf5.read(u1, var_name)
            hdf5.close()

            # aktualisiere die Werte der Normen
            value = my_errornorm(u1,u0,norm_type='L2',degree_rise=deg_rise)
            error_L2L2 += value*value*dt
            error_LinfL2 = max(value, error_LinfL2)
            value = my_errornorm(u1,u0,norm_type='H10',degree_rise=deg_rise)
            error_L2H10 += value*value*dt
        ##################### end zeitliche Iteration
        error_L2H1 = error_L2L2 + error_L2H10
        error_L2H1 = sqrt(error_L2H1)
        error_L2L2 = sqrt(error_L2L2)
        error_L2H10 = sqrt(error_L2H10)
        L2_vec[i] = error_L2L2
        H10_vec[i] = error_L2H10
        Linf_vec[i] = error_LinfL2
        print("N="+str(N0)+"    " + str(error_LinfL2) + "   " + str(error_L2L2) + "   "
            + str(error_L2H10) + "   " + str(error_L2H1))

        # Format:  N0  __  error_LinfL2  __  error_L2L2  __  error_L2H10
        file.write(str(1) + "/" + str(N0) + " & " + str(error_LinfL2) + " & "
            + str(error_L2L2) + " & " + str(error_L2H10) + " \\\\" + "\n")

    file.close()
    convergence_rate(N_vec,L2_vec,H10_vec,Linf_vec,path+var)
