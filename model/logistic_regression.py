import numpy as np 


def sigmoid(x):
    return 1/(1+np.exp(-x))

def compute_W(y, m):
    temp = sigmoid(y * m)
    temp = temp * (1 - temp)
    return np.diag(temp.squeeze())

def compute_alpha(X, W, z, lambda_regularisation):
    n = X.shape[0]
    sqrt_W = np.sqrt(W)
    temp = sqrt_W.dot(X).dot(sqrt_W) + n*lambda_regularisation*np.eye(n)
    temp_inverse = np.linalg.inv(temp)
    alpha = sqrt_W.dot(temp_inverse).dot(sqrt_W).dot(z)
    return alpha

class LogisticRegression:
    def __init__(self, lambda_regularisation=0.01):
        self.lambda_regularisation = lambda_regularisation
        self.alpha_star = None
            
    def fit(self, X, y, alpha=None, epsilon=1):
        """
        fit  using IRLS
        """
        
        if alpha is None:
            alpha = np.random.rand(X.shape[0], 1)

        lambda_regularisation= self.lambda_regularisation
        y.resize([y.shape[0], 1])
        alpha_prev = np.array(alpha)
        m = X.dot(alpha)
        W = np.nan_to_num(compute_W(y, m))
        # z <-- m_i - P_i^T yi / W_i^t = m_i + y_i / sigma(y_i m_i)
        z = m + y/sigmoid(-y * m)
        alpha = compute_alpha(X, W, z, lambda_regularisation)
        
        y.resize([y.shape[0], ])
        if np.linalg.norm(alpha - alpha_prev) > epsilon:
            self.fit(X, y, alpha, epsilon)
        else:
            self.alpha_star = alpha

    def predict(self, X_test):
        # predict prob
        prediction = ((self.alpha_star.T.dot(X_test)).T).reshape(-1)
        prediction= sigmoid(prediction).reshape(-1)
        # predict y
        prediction = np.array(prediction >= 0.5, dtype=int)
        prediction[prediction == 0] = -1
        return prediction