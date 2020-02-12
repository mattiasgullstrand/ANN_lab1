import numpy as np
import activation_functions_v2 as act_funs
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist

class RBF_Net():
    """
        - X is the input matrix where each column is a sample
        - T is the target matrix where coulmn i is the target 
            for the ith coulmn of X
        - mus is a vector of the rbf centres
        - sigmas is a vector of the rbf standard deviations
        - weights is a vector of the weights for the weighted
            sum to compute the output for a given x.
        
    """
    def __init__(self, X, hidden_dim, parties):
        self.N = X.shape[1]
        self.hidden_dim = hidden_dim
        self.D = X.shape[0]
        self.X = X
        self.epochs = 1
        self.mus = self.init_mus()
        self.sigmas = self.init_sigmas()
        self.rbf_step_size = 0.2
        self.parties = parties
        self.neighborhood = 1
    
    def init_mus(self):
        mu_grid = np.zeros([10, 10], dtype = object)
        for i in range(10):
            for j in range(10):
                rand_pattern = np.array(
                    [random.uniform(0, 1) for k in range(self.D)])
                rand_pattern = rand_pattern.reshape(1, self.D)
                mu_grid[i,j] = rand_pattern
        return mu_grid
    
    def init_sigmas(self):
        return np.ones(self.hidden_dim)
    
    def CL_rbf_units(self):
        for pattern_idx in range(self.N):
            x = self.X[:,pattern_idx].reshape(1, self.X.shape[0])
            distances = np.zeros([self.hidden_dim, 2], dtype = object)
            smallest_dist = np.inf
            for i in range(10):
                for j in range(10):
                    mu = self.mus[i,j]
                    dist = self.manhattan(x, mu)
                    if dist < smallest_dist:
                        smallest_dist = dist
                        smallest_i = i
                        smallest_j = j
            winner = self.mus[smallest_i,smallest_j]
            update_is = []
            update_js = []
            for k in range(self.neighborhood):
                i_idx = smallest_i + k
                for l in range(self.neighborhood):
                    j_idx = smallest_j + l
                    if i_idx < 10 and j_idx < 10:
                        update_is.append(i_idx)
                        update_js.append(j_idx)
            for k in range(self.neighborhood):
                i_idx = smallest_i - k
                for l in range(self.neighborhood):
                    j_idx = smallest_j - l
                    if i_idx >= 0 and j_idx >= 0:
                        update_is.append(i_idx)
                        update_js.append(j_idx)
            #print('Winner:', smallest_i, smallest_j)
            for update_i, update_j in zip(update_is, update_js):
                #print(update_i, update_j)
                self.update_rbf_centre(x, update_i, update_j)
    
    def update_rbf_centre(self, x, w_i, w_j):
        w = self.mus[w_i, w_j]
        self.mus[w_i, w_j] = w + self.rbf_step_size*(x - w)
    
    def sq_2_norm(self, x1, x2):
        return np.matmul((x1 - x2).T, (x1 - x2))
    
    def manhattan(self, x1, x2):
        return cdist(x1, x2, metric='cityblock')
    
    def train(self):
        for e in range(self.epochs):
            self.CL_rbf_units()
    
    def get_grid(self):
        grid = np.zeros([10,10])
        smallest_dist = np.inf
        for i in range(10):
            for j in range(10):
                mu = self.mus[i,j]
                for p in range(self.N):
                    x = self.X[:,p].reshape(1, self.X.shape[0])
                    dist = self.manhattan(x, mu)
                    if dist < smallest_dist:
                        smallest_idx = p
                        smallest_dist = dist
                grid[i,j] = self.parties[smallest_idx]
                print(grid)
            
    