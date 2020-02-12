import numpy as np
import activation_functions_v2 as act_funs
import matplotlib.pyplot as plt
import random

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
    def __init__(self, X, T, hidden_dim, weight_learn = 'ls', CL = True, init_mode = 'equal'):
        self.N = X.shape[1]
        self.output_dim = T.shape[0]
        self.hidden_dim = hidden_dim
        self.D = X.shape[0]
        self.X = X
        self.T = T
        self.eta = 0.01
        self.errors = []
        self.epochs = 1
        self.mus = self.init_mus()
        self.sigmas = self.init_sigmas()
        self.weights = self.init_weights()
        self.weight_learn = weight_learn
        self.CL = CL
        self.rbf_step_size = 0.2
        self.rbf_updates = self.N
        self.winners = 1
    
    def init_mus(self):
        if self.init_mode == 'random':
            return np.array([random.uniform(np.min(self.X), np.max(self.X))
                         for i in range(self.hidden_dim)])
        else:
            return np.linspace(np.min(self.X), np.max(self.X), self.hidden_dim)
    
    def init_sigmas(self):
        return np.ones(self.hidden_dim)
    
    def init_weights(self):
        return np.random.rand(self.hidden_dim, 1)
    
    def calc_Phi(self, X):
        # Specify which X
        Phi = np.zeros([self.N, self.hidden_dim])
        for i in range(self.N):
            x = X[:,i].reshape(X.shape[0], 1)
            for j in range(self.hidden_dim):
                mu_j = self.mus[j]
                sigma_j = self.sigmas[j]
                Phi[i,j] = act_funs.gaussian_rbf(x, mu_j, sigma_j)
        return Phi
    
    def learn_weights_least_squares(self):
        # Learn weights in batch mode, taking all samples
        # into consideration per update. Using pseudo-inverse. 
        Phi = self.calc_Phi(self.X)
        f = self.T
        Phi_sq_inv = np.linalg.pinv(np.matmul(Phi.T, Phi))
        Phi_t_f = np.matmul(Phi.T, f)
        self.weights = np.matmul(Phi_sq_inv, Phi_t_f)
    
    def learn_weights_delta_rule(self):
        # Incrementally update the weights with the delta rule
        Phi = self.calc_Phi(self.X)
        f = self.T
        for k in range(self.N):
            Phi_xk = Phi[k,:].reshape(Phi.shape[1], 1)
            f_xk = f[k,:].reshape(1,1)
            e = f_xk - np.matmul(Phi_xk.T, self.weights)
            weight_delta = self.eta * e * Phi_xk
            self.weights = np.add(self.weights, weight_delta)
    
    def batch_learn_weights(self):
        if self.weight_learn == 'delta':
            self.learn_weights_delta_rule()
        else:
            self.learn_weights_least_squares()
        Phi = self.calc_Phi(self.X)
        net_out = np.matmul(Phi, self.weights)
        f = self.T
        abs_model_diff = np.abs(net_out - f)
        abs_residual = np.mean(abs_model_diff)
        self.errors.append(abs_residual)
    
    def CL_rbf_units(self):
        for update in range(self.rbf_updates):
            rand_idx = random.randrange(self.N)
            x = self.X[:,rand_idx].reshape(self.X.shape[0], 1)
            distances = np.zeros([self.hidden_dim, 2], dtype = object)
            for unit_idx in range(self.hidden_dim):
                dist = self.sq_2_norm(x, self.mus[unit_idx])
                distances[unit_idx,:] = np.array([unit_idx, dist])
            sorted_dists = distances[distances[:,1].argsort()]
            unit_inds = sorted_dists[:self.winners,:][:,0]
            for update_idx in unit_inds:
                self.update_rbf_centre(x, update_idx)
    
    def update_rbf_centre(self, x, w_idx):
        w = self.mus[w_idx]
        self.mus[w_idx] = w + self.rbf_step_size*(x - w)
    
    def sq_2_norm(self, x1, x2):
        return np.matmul((x1 - x2).T, (x1 - x2))
        
    def square_transform(self, x):
        return np.sign(x)
    
    def train(self):
        for e in range(self.epochs):
            if self.CL:
                self.CL_rbf_units()
            self.batch_learn_weights()
    
    def predict(self, X):
        Phi = self.calc_Phi(X)
        net_out = np.matmul(Phi, self.weights)
        return net_out
    
    def plot_errors(self):
        plt.plot(self.errors)
        print("Obtained error", self.errors[-1:])
    