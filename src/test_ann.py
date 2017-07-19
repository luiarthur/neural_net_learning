import ann
import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

# Generate a dataset and plot it
np.random.seed(0)
N = 30
X, y = sklearn.datasets.make_moons(N, noise=0.20)
K = X.shape[1]

### Plot Data
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()


nnodes = [7,2]
H = len(nnodes)
W = [None] * H
b = [None] * H

for h in range(H):
    nnode_prev = K if h == 0 else nnodes[h-1]
    W[h] = np.asmatrix(np.random.randn(nnode_prev,nnodes[h]))
    b[h] = np.asmatrix(np.random.randn(nnodes[h]))

reload(ann)
y_hat = ann.forward(X, W, b)

ann.predict(y_hat)
ann.loss(y_hat, y)
