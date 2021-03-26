import numpy as np
from scipy.spatial.distance import cdist
# local import
from model.spectrum import compute_kernel, get_all_combinations_letters, get_sequences_distributions_one_mismatch, get_sequences_distributions_spectrum
from model.substring import substring_rec
from model.LA import LA_kernel,compute_substitution_matrix
from util.config import LETTERS, ALPHABET


class Kernel:
    def __init__(self, kernel_name, params):
        self.name = kernel_name
        self.params = params
        # for spectrum and mismatch
        self.X_train_repres = None
        self.X_test_repres = None

    def get_kernel_func(self, X1):
        """
        return kernel function k(x, y) according to kernel picked
        """
        n11, n12 = np.shape(X1)
        if self.name == "linear":
            def k(x, y):
                return np.inner(x, y)
        elif self.name == "poly":
            offset = self.params["poly_offset"]
            dim = self.params["dim"]
            def k(x,y):
                return (offset + np.dot(x, y))**dim
        elif self.name == "sum":
            # TODO marche pas trop
            offset = self.params["poly_offset"]
            sigma = self.params["sigma"]
            if  sigma == 0:
                gamma = 1.0 / (n12 * X1.var())
            else:
                gamma = 1.0 / (2.0 * sigma ** 2)
            if gamma == 0:
                gamma = 1/(self.params["dim"]*self.params["sigma"]**2)
            dim = self.params["dim"]
            def k(x,y):
                return (offset + np.dot(x, y))**dim + np.inner(x, y) + np.exp(- gamma*np.linalg.norm(x - y)**2)
        elif self.name == "rbf":
            sigma = self.params["sigma"]
            if  sigma == 0:
                gamma = 1.0 / (n12 * X1.var())
            else:
                gamma = 1.0 / (2.0 * sigma ** 2)
            def k(x, y):
                return np.exp(-gamma*np.linalg.norm(x-y)**2 )


        return k

    def compute_mat_by_coeff(self, X1, X2):
        """
        generic function to compute kernel matrix K with function k(x, y)
        """
        k = self.get_kernel_func(X1)
        K = np.zeros((X2.shape[0], X1.shape[0]))
        for i in range(X2.shape[0]):
            for j in range(X1.shape[0]):
                K[i,j] = k(X2[i], X1[j])
        return K

    def get_kernel_mat(self, X1, X2, test = False, X_test_repres=None, X_train_repres=None):
        """
        compute kernel matrix for various kernels
        X_test_repres and X_train_repres are optional arguments to not re calculate X_test and X_train 
        representation each time.
        """
        print("## computing kernel")
        if self.name == "rbf":
            K = self.compute_gaussian_kernel(X1, X2)
        elif self.name == "spectrum":
            K =  self.compute_spectrum_kernel(X1,X2, test, X_test_repres, X_train_repres)
        elif self.name == "mismatch":
            K =  self.compute_mismatch_kernel(X1,X2, test, X_test_repres, X_train_repres)
        elif self.name == "substring":
            K =  self.compute_substring_kernel(X1,X2)
        elif self.name == "LA":
            K =  self.compute_LA_kernel(X1,X2)
        else:
            K = self.compute_mat_by_coeff(X1, X2)
        return K

    def compute_gaussian_kernel(self, X1, X2):
        """
        compute rbf kernel matrix faster
        """
        _,n12 = np.shape(X1)
        sigma = self.params["sigma"]
        if  sigma == 0:
            gamma = 1.0 / (n12 * X1.var())
        else:
            gamma = 1.0 / (2.0 * sigma ** 2)

        dist = cdist(X2, X1, metric='sqeuclidean')
        K = np.exp(-gamma * dist)
        return K


    def compute_spectrum_kernel(self, X1, X2, test, X_test_repres=None, X_train_repres=None):
        """
        computes spectrum kernel matrix
        """
        k_mers = self.params["k_mers"]
        if X_train_repres is None:
            all_combinations_letters = get_all_combinations_letters(k_mers)
            print("## length features spectrum", len(all_combinations_letters))
            X_train_repres = np.empty([len(X2), len(all_combinations_letters)])
            for i, row in enumerate(X2):
                X_train_repres[i,:] = get_sequences_distributions_spectrum(row, all_combinations_letters, k_mers)

        if test == False: # train
            K = compute_kernel(X_train_repres)
        else: # test
            if X_test_repres is None:
                all_combinations_letters = get_all_combinations_letters(k_mers)
                X_test_repres= np.empty([len(X1), len(all_combinations_letters)])
                for i, row in enumerate(X1):
                    X_test_repres[i,:] = get_sequences_distributions_spectrum(row, all_combinations_letters, k_mers)
            K = compute_kernel(X_train_repres, X_test_repres)
        return K


    def compute_mismatch_kernel(self, X1, X2, test, X_test_repres=None, X_train_repres=None):
        """
        computes mismatch kernel matrix
        """
        k_mers = self.params["k_mers"]
        if X_train_repres is None:
            all_combinations_letters = get_all_combinations_letters(k_mers)
            print("## length features mismatch", len(all_combinations_letters))
            X_train_repres = np.empty([len(X2), len(all_combinations_letters)])
            for i, row in enumerate(X2):
                X_train_repres[i,:] = get_sequences_distributions_one_mismatch(row, all_combinations_letters, k_mers)

        if test == False: # train
            K = compute_kernel(X_train_repres)
        else: # test
            if X_test_repres is None:
                all_combinations_letters = get_all_combinations_letters(k_mers)
                X_test_repres= np.empty([len(X1), len(all_combinations_letters)])
                for i, row in enumerate(X1):
                    X_test_repres[i,:] = get_sequences_distributions_one_mismatch(row, all_combinations_letters, k_mers)
            K = compute_kernel(X_train_repres, X_test_repres)
        return K


    def compute_substring_kernel(self, X1, X2):
        """
        compute substring kernel matrix
        """
        K = np.zeros((X2.shape[0], X1.shape[0]))
        n = X2.shape[0]
        for i, row1 in enumerate(X2):
            count = 0
            for j, row2 in enumerate(X1):
                K[i,j] = substring_rec(row1, row2, 0.1, 1)
                count += 1
                if count%100==0:
                    print('######## STEP {}: {}/{}########'.format(i,count,n))
        return K


    def compute_LA_kernel(self, X1, X2,test=False):
        """
        compute kernel matric for local alignment kernel
        """
        if test:
            S = compute_substitution_matrix(X2)
        else:
            S = compute_substitution_matrix(X1)
        print(S)
        print('Successfull computation of the substitution matrix')
        K = np.zeros((X2.shape[0], X1.shape[0]))
        n = X2.shape[0]
        for i, row1 in enumerate(X2):
            count = 0
            for j, row2 in enumerate(X1):
                K[i,j] = LA_kernel(S,beta=0.05,d=1,e=11,x=row1,y=row2)
                count += 1
                if count%100==0:
                    print('######## STEP {}: {}/{}########'.format(i,count,n))
        return K
