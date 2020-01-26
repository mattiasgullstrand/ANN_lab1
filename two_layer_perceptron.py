import numpy as np
import activation_functions as act_funs
import matplotlib.pyplot as plt

class Two_Layer_Delta_Rule():
    """ Using cost function 0.5 * ∑_k (t_k − y_k)^2
    """
    def __init__(self, X, T, hidden_dim, bias = True):
        self.N = X.shape[1]
        self.bias = bias
        self.output_dim = T.shape[0]
        self.hidden_dim = hidden_dim
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
        self.W1_init = self.init_W1() # Store initial W1
        self.W1_train = self.W1_init
        self.W2_init = self.init_W2() # Store initial W2
        self.W2_train = self.W2_init
        #self.act_fun = Add activation function
    
    def X_add_bias(self, X):
        X_bias = np.ones([self.D, self.N])
        X_bias[:-1,:] = np.copy(X)
        return X_bias
    
    def init_W1(self):
        return np.random.rand(self.hidden_dim, self.D)
    
    def init_W2(self):
        return np.random.rand(self.output_dim, self.hidden_dim)
    
    def seq_learn(self):
        se = 0
        delta_j_vec = np.zeros(self.hidden_dim)
        delta_k_vec = np.zeros(self.output_dim)
        
        for n in range(self.N):
            delta_W2 = np.zeros([self.output_dim, self.hidden_dim])
            delta_W1 = np.zeros([self.hidden_dim, self.D])
            x = self.X[:,n].reshape(self.X.shape[0], 1)
            h_in = np.matmul(self.W1_train, x)
            h = act_funs.shifted_sigmoid(h_in)
            y_in = np.matmul(self.W2_train, h)
            y = act_funs.shifted_sigmoid(y_in)
            t = self.T[:,n]
            for j in range(self.hidden_dim):
                for k in range(self.output_dim):
                    t_k = t[k]
                    y_k_in = y_in[k]
                    act_fun_prime = act_funs.shifted_sigmoid_prime(y_k_in)
                    delta_k_vec[k] = (t_k - y[k])*act_fun_prime
                    delta_W2[k,j] += self.eta*delta_k_vec[k]*h_in[j]
            
            for i in range(self.D):
                for j in range(self.hidden_dim):
                    w_j = self.W2_train[:,j] # len(w_j) = k
                    h_j_in = h_in[j]
                    act_fun_prime = act_funs.shifted_sigmoid_prime(h_j_in)
                    delta_j_vec[j] = np.sum(np.matmul(delta_k_vec, w_j)*
                                           act_fun_prime)
                    delta_W1[j,i] += self.eta*delta_j_vec[j]*x[i]
            
            e = t - y
            se += 0.5*(e)**2
        
            self.W2_train = np.add(self.W2_train, delta_W2)
            self.W1_train = np.add(self.W1_train, delta_W1)
        mse = np.mean(se)
        self.errors.append(mse)
    
    def batch_learn(self):
        se = 0
        delta_j_vec = np.zeros(self.hidden_dim)
        delta_k_vec = np.zeros(self.output_dim)
        delta_W2 = np.zeros([self.output_dim, self.hidden_dim])
        delta_W1 = np.zeros([self.hidden_dim, self.D])
        
        for n in range(self.N):
            x = self.X[:,n].reshape(self.X.shape[0], 1)
            h_in = np.matmul(self.W1_train, x)
            h = act_funs.shifted_sigmoid(h_in)
            y_in = np.matmul(self.W2_train, h)
            y = act_funs.shifted_sigmoid(y_in)
            t = self.T[:,n]
            for j in range(self.hidden_dim):
                for k in range(self.output_dim):
                    t_k = t[k]
                    y_k_in = y_in[k]
                    act_fun_prime = act_funs.shifted_sigmoid_prime(y_k_in)
                    delta_k_vec[k] = (t_k - y[k])*act_fun_prime
                    delta_W2[k,j] += self.eta*delta_k_vec[k]*h_in[j]
            
            for i in range(self.D):
                for j in range(self.hidden_dim):
                    w_j = self.W2_train[:,j] # len(w_j) = k
                    h_j_in = h_in[j]
                    act_fun_prime = act_funs.shifted_sigmoid_prime(h_j_in)
                    delta_j_vec[j] = np.sum(np.matmul(delta_k_vec, w_j)*
                                           act_fun_prime)
                    delta_W1[j,i] += self.eta*delta_j_vec[j]*x[i]
            
            e = t - y
            se += 0.5*(e)**2
        
        self.W2_train = np.add(self.W2_train, delta_W2)
        self.W1_train = np.add(self.W1_train, delta_W1)
        mse = np.mean(se)
        self.errors.append(mse)
    
    def train(self, learn_type = 'batch'):
        for e in range(self.epochs):
            if learn_type == 'batch':
                self.batch_learn()
            elif learn_type == 'seq':
                self.seq_learn()
            else:
                print("Unknown learning type")
    
    def classification_function(self, input_array, sign = True): 
        # Remember to include a 1 for the bias in the input_array
        H_in = np.matmul(self.W1_train, input_array)
        H = act_funs.shifted_sigmoid(H_in)
        y_in = np.matmul(self.W2_train, H)
        y = act_funs.shifted_sigmoid(y_in)
        if sign:
            return np.sign(y)
        else:
            return y
    
    def plot_errors(self):
        plt.plot(self.errors)
        print("Obtained error", self.errors[-1:][0])
    
    def plot_decision_boundary(self): # Works only in two dimensions
        data = self.X
        labels = self.T[0].tolist()
        label_color_map = {
            1: 'b',
            0: 'r',
            -1: 'r'
        }
        label_colors = list(map(lambda x: label_color_map.get(x), labels))

        n_grid = 100
        classify = np.vectorize(lambda x1, x2: self.classification_function(np.array([x1, x2, 1])))
        grid_x, grid_y = np.meshgrid(np.linspace(-2, 2, n_grid),
                                     np.linspace(-1, 6, n_grid))
        grid_class = classify(grid_x, grid_y).flatten()
        grid_colors = np.vectorize(lambda x: label_color_map.get(x))(grid_class)
        plt.scatter(grid_x, grid_y, c=grid_colors, alpha=0.05)
        plt.scatter(data[0, :], data[1, :], c=label_colors)