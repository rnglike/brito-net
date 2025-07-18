import numpy as np

# Activation functions and derivatives
def tansig(x):
    return np.tanh(x)

def dtansig(x):
    return 1.0 - np.tanh(x)**2

def purelin(x):
    return x

def dpurelin(x):
    return np.ones_like(x)