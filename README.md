# cahn-hilliard-tumour-fenics
A Finite Element Approximation of a Cahn--Hilliard Tumour Model with FEniCS,
by Dennis Trautwein (2020).

DOI of the original tumour model of Garcke, Lam, Sitka and Styles:

https://doi.org/10.1142/S0218202516500263

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# General information about FEniCS:

a) Installation of FEniCS:
See e.g. the following link:

 https://fenicsproject.org/download/


b) Running FEniCS:
Run a FEniCS program with the command:

> python3 [name].py


c) Visualization:
VTK-/VTU-/PVD- output files can be visualized with the program PARAVIEW. 
See e.g. the following link for download and installation:

 https://www.paraview.org/download/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Directories:

1) The program 'ch_tumour_simple.py' is a simplified version of our programs. 
It is a modification of the demo 'demo_cahn-hilliard.py' from the FEniCS homepage:

 https://fenicsproject.org/docs/dolfin/latest/python/demos.html


2) The directory 'program_1D_convergence' contains the programs solving the 
Cahn--Hilliard tumour system and computing convergence rates in one dimension
on the interval (0,1). 


3) The directory 'program_2D' contains the programs solving the Cahn--Hilliard
tumour system in two dimensions on the square (0,12.5)^2.


4) The directory 'program_3D' contains the programs solving the Cahn--Hilliard 
tumour system in three dimensions on the cube (0,3)^3.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Programs:

i) In each directory you can run the program with the command:

> python3 main_program.py


ii) The file 'settings.py' contains all settings: 
Parameters for the finite element discretization (mesh size, polynom degree, quadrature degree etc.) 
and other settings, e.g. for saving the output (formats like .vtu, .xml, .hdf4), handling initial 
values from a file, etc.


iii) The file 'class_Tumour.py' contains all model parameters and initializes the 
weak formulation of the system of partial differential equations.


iv) The file 'useful_functions.py' contains all functions needed for the program.


v) The file 'main_convergence_rate.py' (see the directory 'program_1D_convergence') can
calculate the errors in some specific norms and the experimental orders of convergence (EOC), 
and save them into files. You can run the program with the command:

> python3 main_convergence_rate.py

Note: Before running 'main_convergence_rate.py', you have to generate the data of solutions with different mesh sizes 'h' and you have to choose a reference solution. 
You can change the mesh size, polynom degree, quadrature degree in 'settings.py'.
