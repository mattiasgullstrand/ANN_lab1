import numpy as np
import matplotlib.pyplot as plt

n = 100
mA = np.array([2, 1])
mB = np.array([-1, 0])
sigmaA = 0.5
sigmaB = 0.5
classA = np.zeros([n,2])
classB = np.zeros([n,2])
classA[:,0] = np.random.normal(mA[0], sigmaA, n)
classA[:,1] = np.random.normal(mA[1], sigmaA, n)
classB[:,0] = np.random.normal(mB[0], sigmaB, n)
classB[:,1] = np.random.normal(mB[1], sigmaB, n)

plt.plot(classA[:,0], classA[:,1], 'bo', color = 'b')
plt.plot(classB[:,0], classB[:,1], 'bo', color = 'r')

data = np.concatenate((classA, classB), axis=0)
labels_delta = np.array([1]*n + [-1]*n)

data_labels = np.concatenate((data, labels_delta.reshape(n*2,1)), axis=1)
np.random.shuffle(data_labels)

X = data_labels[:,:2].T
T_delta = data_labels[:,2].reshape(n*2, 1).T

T_percept = T_delta
for i, t in enumerate(T_percept[0,:]):
    if t == -1:
        T_percept[0,i] = 0

W = np.random.rand(T_percept.shape[0],X.shape[0])
W_bias = np.vstack([W, np.ones([1,W.shape[1]])])

def indicator(W, x):
    if np.matmul(W,x) >= 0:
        return 1
    else:
        return 0

lr = 0.001
epoch = 1
for e in range(epoch):
    for i in range(X.shape[1]):
        x = X[:,i].reshape(X.shape[0], 1)
        y = indicator(W,x)
        t = T_percept[:,i]
        delta_W = np.zeros(W.shape)
        if t == 0 and y == 1:
            delta_W = -lr*x.T
        if t == 1 and y == 0:
            delta_W = lr*x.T
        W = np.add(W, delta_W)
        #print(W)

x_val = np.array([-2.5,-2]).reshape(2,1)
def norm_vec(W):
    N = np.zeros(W.shape)
    N[:,0] = -1
    N[:,1] = W[:,0]/W[:,1]
    return N.T

plt.plot(classA[:,0], classA[:,1], 'bo', color = 'b')
plt.plot(classB[:,0], classB[:,1], 'bo', color = 'r')
plt.plot(np.dot(norm_vec(W), x_val.T))
plt.show()