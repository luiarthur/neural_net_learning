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

    y_hat = map(lambda yi: softmax(yi), np.asarray(Z[-1]))
    return np.asmatrix(y_hat)

def loss(y_hat, y):
    N = len(y)
    assert y_hat.shape[0] == N, "Required: y_hat.shape[0] == len(y)"

    C = y_hat.shape[1] # Number of classes
    y_true_mat = np.zeros( (N,C) )
    for n in range(N):
        y_true_mat[n, y[n]] = 1

    loss = -np.sum(y_true_mat * np.log(np.asarray(y_hat)))

    return loss / N

def predict(y_hat):
    return map(lambda yi: np.argmax(yi), y_hat.tolist())

