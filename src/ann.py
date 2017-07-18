import numpy as np

def logit(p):
    # maps value in unit interval (p) to real line
    return np.log(p / (1.0-p))

def logistic(x):
    # maps value in real line (x) to unit interval
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    mx = max(x)
    y = np.exp(x - mx)
    return y / sum(y)

def forward(X, W, b, act_fn=logistic):
    N = X.shape[0]
    H = len(b)
    assert len(W) == len(b), "Required: len(W) == len(b)"
    Z = [None] * H
    
    for h in range(H):
        # Input Matrix:
        M = X if h == 0 else act_fn(Z[h-1])
        Z[h] = M * W[h] + np.kron(np.ones((N,1)), b[h])

    y = map(lambda yi: softmax(yi), np.asarray(Z[-1]))
    return np.asmatrix(y)

