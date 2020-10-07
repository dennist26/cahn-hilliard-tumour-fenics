from dolfin import *
from settings import *
import random
import numpy as np


''' functions for initial conditions '''

def function_5(x, model):
    if x[0]<DOLFIN_EPS:
        phi = np.pi/2
    else:
        phi = np.arctan(x[1]/x[0])
    r = 0
    for i in range(0,len(x)):
        r += x[i]**2
    r = r**(1./2)
    r += -(2+0.1*cos(2*phi))
    value = -np.tanh(r/(sqrt(2)*model.epsilon))
    return value
