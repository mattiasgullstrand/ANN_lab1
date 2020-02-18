import numpy as np

def shifted_sigmoid(x):
    return 2/(1 + np.exp(-x)) - 1

def shifted_sigmoid_prime(x):
    return 0.5*np.multiply((1 + shifted_sigmoid(x)), (1 - shifted_sigmoid(x)))

def gaussian_rbf(x, mu, sigma):
    np.exp(-np.square(x - mu)/(2*np.square(sigma)))