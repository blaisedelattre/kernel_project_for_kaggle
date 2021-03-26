from model.logistic_regression import LogisticRegression
from model.svm import SVM
from os import path
from model.kernel import Kernel
from sklearn import svm
# local import
from util.config import *
from util.util import dump_object, load_object

class Model:

    def __init__(self, classifier_type, classifier_params, kernel_name, kernel_params, id_model=0):
        
        # define kernel
        self.kernel = Kernel(kernel_name, kernel_params)
        self.tag_kernel = str(id_model) + "_kernel_"+ kernel_name # kernel tag used to kernel mat saves
        
        # define classifier 
        if classifier_type == "svm":
            self.classifier = SVM(classifier_params)
        elif classifier_type == "l_regression":
            self.classifier = LogisticRegression(lambda_regularisation=0.01)
        
        # kernel mat
        self.kernel_mat_train = None
        self.kernel_mat_val = None
        self.kernel_mat_test = None
        
        # load save kernel matrix 
        self.do_save_kernel = kernel_params["save_kernel"] 
        self.save_name = kernel_params["save_name"]
        if kernel_params["load_kernel"]:
            self.kernel_mat_train = load_object("train_" + self.tag_kernel + "_" + kernel_params["load_name"])
            self.kernel_mat_val = load_object("val_" + self.tag_kernel + "_" + kernel_params["load_name"])
            self.kernel_mat_test = load_object("test_" + self.tag_kernel + "_" + kernel_params["load_name"])
            if self.kernel_mat_train is None: 
                print("## kernel load failed: kernel not found")
            else:
                print("## kernel matrix loaded")
            
    def fit(self, X, y, X_train_repres=None):
        """
        fit the model 
        use X_train_repres if filled as it is faster
        """
        self.X_train = X
        self.deal_with_kernel_mat_train(X_train_repres)
        self.classifier.fit(self.kernel_mat_train, y)
        

    def predict(self, X_test, X_test_repres=None, X_train_repres=None):
        """
        predict from X_test
        for spectrum and mismatch only
            X_test_repres : representation of X_test in RKHS
            X_train_repres : representation of X_train in RKHS
            use X_train_repres, X_test_repres if filled as it is faster 
        """
        self.deal_with_kernel_mat_test(X_test, X_test_repres, X_train_repres)
        return self.classifier.predict(self.kernel_mat_test)

    def predict_val(self, X_val, X_val_repres=None, X_train_repres=None):
        """
        predict from X_test
        for spectrum and mismatch only
            X_val_repres : representation of X_val in RKHS
            X_train_repres : representation of X_train in RKHS
            use X_train_repres, X_test_repres if filled as it is faster 
        """
        self.deal_with_kernel_mat_val(X_val, X_val_repres, X_train_repres)
        return self.classifier.predict(self.kernel_mat_val)

    def predict_train(self):
        """
        to compute the train loss
        """
        return self.classifier.predict(self.kernel_mat_train)

    ## functions to calculate or load Kernel matrix and save it if needed  
    def deal_with_kernel_mat_train(self, X_train_repres):
        """
        compute kernel matrix and save it if do_save_kernel
        """
        if self.kernel_mat_train is None:
            # compute kernel
            self.kernel_mat_train = self.kernel.get_kernel_mat(self.X_train, self.X_train, X_train_repres=X_train_repres)
            # save it or not
            dump_object(self.kernel_mat_train, "train_" + self.tag_kernel + "_" + self.save_name , self.do_save_kernel)

    def deal_with_kernel_mat_test(self, X_test, X_test_repres=None, X_train_repres=None):
        """
        compute kernel matrix and save it if do_save_kernel
        """
        if self.kernel_mat_test is None:
            # compute kernel
            self.kernel_mat_test = self.kernel.get_kernel_mat(X_test, self.X_train, test=True, X_test_repres=X_test_repres, X_train_repres=X_train_repres)
            #  save it or not
            dump_object(self.kernel_mat_test, "test_" + self.tag_kernel + "_" + self.save_name , self.do_save_kernel)

    def deal_with_kernel_mat_val(self, X_val, X_val_repres, X_train_repres=None): 
        """
        compute kernel matrix and save it if do_save_kernel
        """
        if self.kernel_mat_val is None:
            # compute kernel
            self.kernel_mat_val = self.kernel.get_kernel_mat(X_val, self.X_train, test=True, X_test_repres=X_val_repres, X_train_repres=X_train_repres)
            #  save it or not
            dump_object(self.kernel_mat_val, "val_" + self.tag_kernel + "_" + self.save_name , self.do_save_kernel)


    def set_c_svm(self, c_svm):
        """
        set c_svm in SVM classifier
        """
        self.classifier.C = c_svm
        return self