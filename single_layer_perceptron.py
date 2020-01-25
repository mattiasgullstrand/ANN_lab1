import numpy as np
import activation_functions as act_funs
import matplotlib.pyplot as plt

class Single_Layer_Perceptron():
    
    def __init__(self, X, T, bias = True):
        self.N = X.shape[1]
        self.bias = bias
        self.output_dim = T.shape[0]
        if bias == True:
            self.D = X.shape[0] + 1
            self.X = self.X_add_bias(X)
        else:
            self.D = X.shape[0]
            self.X = X
        self.T = T
        self.eta = 0.001
        self.errors = []
        self.epochs = 100
        self.W_init = self.init_W() # Store initial W
        self.W_train = self.W_init
        self.converge_epoch = np.inf
    
    def X_add_bias(self, X):
        X_bias = np.ones([self.D, self.N])
        X_bias[:-1,:] = np.copy(X)
        return X_bias
    
    def init_W(self):
        return np.random.rand(self.output_dim, self.D)
    
    def indicator(self,W,x):
        if np.matmul(W,x) >= 0:
            return 1
        else:
            return 0
    
    def seq_learn(self):
        se = 0
        for i in range(self.N):
            delta_W = 0
            x = self.X[:,i].reshape(self.D, 1)
            y = self.indicator(self.W_train, x)
            t = self.T[:,i]
            if t == 0 and y == 1:
                delta_W = -self.eta*x.T
            if t == 1 and y == 0:
                delta_W = self.eta*x.T
            self.W_train = np.add(self.W_train, delta_W)
            se += (y-t)**2
        mse = np.mean(se)
        self.errors.append(mse)
    
    def batch_learn(self):
        se = 0
        delta_W = 0
        for i in range(self.N):
            x = self.X[:,i].reshape(self.D, 1)
            y = self.indicator(self.W_train, x)
            t = self.T[:,i]
            if t == 0 and y == 1:
                delta_W += -self.eta*x.T
            if t == 1 and y == 0:
                delta_W += self.eta*x.T
            se += (y-t)**2
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