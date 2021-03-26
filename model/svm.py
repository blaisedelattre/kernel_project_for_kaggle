import numpy as np
from cvxopt import solvers
solvers.options['show_progress'] = False
from cvxopt import matrix, spmatrix
import cvxopt


class SVM:
    def __init__(self, params):
        self.C = params["c_svm"]
        self.alpha_star = None
        self.eps = 1E-5


    def fit(self, X, y):
        """
        compute C-SVM with dual
        """
        print("## training")
        n = X.shape[0]
        diag_y = np.diag(y.squeeze())
        P = cvxopt.matrix(X, tc='d')
        q = cvxopt.matrix(-y, tc='d')
        h =  cvxopt.matrix(np.vstack( (self.C*np.ones((n,1)), np.zeros((n,1))) ), tc='d')
        G = cvxopt.matrix(np.concatenate((diag_y, -diag_y), axis=0), tc='d')
        sol = cvxopt.solvers.qp(P, q, G, h)
        self.alpha_star = np.ravel(sol['x']).squeeze()
        # get vector supports
        self.alpha_star[np.where(np.abs(self.alpha_star) < self.eps)] = 0
        self.biais = np.array(np.mean(y - np.dot(self.alpha_star, X))).squeeze()


    def predict(self, X):
        """
        compute prediction
        """
        return np.sign(np.dot(self.alpha_star, X)+ self.biais)
