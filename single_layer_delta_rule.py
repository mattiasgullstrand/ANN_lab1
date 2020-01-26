import numpy as np
import activation_functions as act_funs
import matplotlib.pyplot as plt

class Single_Layer_Delta_Rule():
    
    def __init__(self, X, T, bias = True):
        self.N = X.shape[1]
        self.bias = bias
        self.output_dim = T.shape[0]
        if bias:
            self.D = X.shape[0] + 1
            self.X = self.X_add_bias(X)
        else:
            self.D = X.shape[0]
            self.X = X
        self.T = T
        self.eta = 0.001
        self.errors = []
        self.epochs = 1
        self.W_init = self.init_W() # Store initial W
        self.W_train = self.W_init
        self.converge_epoch = np.inf
    
    def X_add_bias(self, X):
        X_bias = np.ones([self.D, self.N])
        X_bias[:-1,:] = np.copy(X)
        return X_bias
    
    def init_W(self):
        return np.random.rand(self.output_dim, self.D)
    
    def seq_learn(self):
        se = 0
        for i in range(self.N):
            x = self.X[:,i].reshape(self.X.shape[0], 1)
            y = np.matmul(self.W_train, x)
            t = self.T[:,i]
            e = t - y
            delta_W = self.eta*e*x.T
            self.W_train = np.add(self.W_train, delta_W)
            se += 0.5*(e)**2
        mse = np.mean(se)
        self.errors.append(mse)
    
    def batch_learn(self):
        se = 0
        delta_W = 0
        for i in range(self.N):
            x = self.X[:,i].reshape(self.X.shape[0], 1)
            y = np.matmul(self.W_train, x)
            t = self.T[:,i]
            e = t - y
            delta_W += self.eta*e*x.T
            se += 0.5*(e)**2
        mse = np.mean(se)
        self.W_train = np.add(self.W_train, delta_W)
        self.errors.append(mse)
    
    def train(self, learn_type = 'batch'):
        for e in range(self.epochs):
            if learn_type == 'batch':
                self.batch_learn()
            elif learn_type == 'seq':
                self.seq_learn()
            else:
                print("Unknown learning type")
    
    def plot_errors(self):
        plt.plot(self.errors)
        print("Obtained error", self.errors[-1:][0])
    
    def plot_decision_boundary(self):
        data = self.X
        labels = self.T[0].tolist()
        weights = self.W_train[0].tolist()
        label_color_map = {
            1: 'b',
            0: 'r',
            -1: 'r'
        }
        label_colors = list(map(lambda x: label_color_map.get(x), labels))
    
        boundary_func = lambda x: -(weights[0]*x + weights[2])/weights[1]
        boundary_x = [-1, 0, 1]
        boundary_y = list(map(boundary_func, boundary_x))

        n_grid = 100
        classify = np.vectorize(lambda x, y: np.sign(weights[0]*x +
                                                     weights[1]*y +
                                                     weights[2]))
        grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, n_grid),
                                     np.linspace(-1, 6, n_grid))
        grid_class = classify(grid_x, grid_y).flatten()
        grid_colors = np.vectorize(lambda x: label_color_map.get(x))(grid_class)
        plt.scatter(grid_x, grid_y, c=grid_colors, alpha=0.05)
        plt.scatter(data[0, :], data[1, :], c=label_colors)
        plt.plot(boundary_x, boundary_y, c='black')
        plt.quiver(boundary_x[1], boundary_y[1], weights[0], weights[1],
                   angles='xy', minlength=1)