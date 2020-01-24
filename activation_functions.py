import numpy as np

def shifted_sigmoid(x):
    return 2/(1 + np.exp(-x)) - 1

def shifted_sigmoid_prime(x):
    return (1 + shifted_sigmoid(x))*(1 - shifted_sigmoid(x))/2