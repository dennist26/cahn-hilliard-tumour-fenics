from dolfin import *
from settings import *
import random
import numpy as np


''' functions for initial conditions '''


''' l^p sphere '''
def function_1(x, model, radius):
    r = 0
    for i in range(0,len(x)):
        r += (x[i]-0.5)**4
    r = r**(1./4)
    r = r/discr.length
    value = - np.tanh((r-radius)/sqrt(2)/model.epsilon)
    return value
