import numpy as np
import activation_functions_v2 as act_funs
import matplotlib.pyplot as plt

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
    def __init__(self, X, T, hidden_dim):
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
    
    def init_mus(self):
        return np.linspace(np.min(self.X), np.max(self.X), self.hidden_dim)
    
    def init_sigmas(self):
        return np.ones(self.hidden_dim)
    
    def init_weights(self):
        return np.random.rand(self.hidden_dim, self.output_dim)
    
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
        # into consideration per update. 
        Phi = self.calc_Phi(self.X)
        f = self.T
        Phi_sq_inv = np.linalg.pinv(np.matmul(Phi.T, Phi))
        Phi_t_f = np.matmul(Phi.T, f)
        self.weights = np.matmul(Phi_sq_inv, Phi_t_f)
    
    def batch_learn_weights(self):
        self.learn_weights_least_squares()
        Phi = self.calc_Phi(self.X)
        net_out = np.matmul(Phi, self.weights)
        f = self.T
        abs_model_diff = np.abs(net_out - f)
        abs_residual = np.mean(abs_model_diff)
        self.errors.append(abs_residual)
        
    def square_transform(self, x):
        return np.sign(x)
    
    def train(self):
        for e in range(self.epochs):
            self.batch_learn_weights()
    
    def predict(self, X):
        Phi = self.calc_Phi(X)
        net_out = np.matmul(Phi, self.weights)
        return net_out
    
    def plot_errors(self):
        plt.plot(self.errors)
        print("Obtained error", self.errors[-1:])
    