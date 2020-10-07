from dolfin import *
from settings import *
import random
import numpy as np


''' functions for initial conditions '''


''' 3D '''
def function_3D(x, model):
    r = 0
    r = 2*abs(x[0])
    r+= 0.5*abs(x[1])
    r+= abs(x[2])
    #r = r**(1./2)
    r = r/discr.length
    value = -np.tanh((r-0.2)/(sqrt(2)*model.epsilon))
    return value


''' 3D '''
def function_3D_2(x, model):
    ''' Polarkoordinaten U'''
    r = sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    if r>0:
        theta = np.arccos(x[2]/r)
    else:
        theta=0
    phi = np.arctan2(x[1],x[0])
    r = r/discr.length
    r+= 0.04*cos(2*theta)
    value = -np.tanh((r-0.1)/(sqrt(2)*model.epsilon))
    return value

''' 3D '''
def function_3D_3(x, model):
    ''' Polarkoordinaten U'''
    r = (0.5*x[0]**4 + 1.5*x[1]**4 + 1.5*x[2]**4)**(1/4)
    r = r/discr.length
    value = -np.tanh((r-0.1)/(sqrt(2)*model.epsilon))
    return value

''' 3D '''
def function_3D_4(x, model):
    ''' Polarkoordinaten U'''
    r = sqrt(0.5*x[0]**2 + 2*x[1]**2 + 0.75*x[2]**2)
    r = r/discr.length
    value = -np.tanh((r-0.1)/(sqrt(2)*model.epsilon))
    return value
