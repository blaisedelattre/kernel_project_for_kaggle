import numpy as np
import argparse
from sklearn.model_selection import KFold
# local import
from model.model import Model
from util.config import *
from util.util import *
from model.spectrum import get_repres_spectrum_kernel, get_repres_mismatch_kernel

ks = [0,1,2]
train_sequences, train_labels, test_sequence = get_raw_train_data(ks)
# classifier
classifier_params = {}
classifier_type = "svm"
classifier_params["c_svm"] = 1 # default value
# kernel 
kernel_name = "spectrum"
kernel_params = {}
kernel_params["load_kernel"] = True
kernel_params["save_kernel"] = True
# hyperparameters grid
k_merss = [3, 4, 5, 6, 7] # 8 prend trop de temps
c_svms = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]
# launch training
n_splits = 5
kfold = KFold(n_splits=n_splits, random_state=42)
val_accuracies = np.zeros((len(k_merss), len(c_svms), n_splits, 3))

for train_sequence, train_label, test_sequence, k in zip(train_sequences, train_labels, test_sequence, ks):
    print("\n## data set " + str(k))
    for idx_fold, (train_index, test_index) in enumerate(kfold.split(train_sequence, train_label)):
        print("\n## fold", idx_fold)
        X_train, y_train = train_sequence[train_index], train_label[train_index]
        X_test, y_test = train_sequence[test_index], train_label[test_index] 
        for i_nn, k_mers in enumerate (k_merss):
            print("\n## k_mers", k_mers)
            kernel_params["k_mers"] = k_mers
            kernel_params["save_name"] = "last_kernel" + "_n_" + str(k_mers) + "_f_" + str(idx_fold)
            kernel_params["load_name"] = kernel_params["save_name"]
            # define model
            model = Model(classifier_type, classifier_params, kernel_name, kernel_params, id_model=str(k))
            for i_csvm, c_svm in enumerate(c_svms):
                print("\n## c svm", c_svm)
                model.set_c_svm(c_svm)
                # fitting
                model.fit(X_train, y_train)
                # train accuracy
                train_accuracy = compute_accuracy(model.predict_train(), y_train)
                print("## train accuracy", train_accuracy)
                # val accuracy
                val_predictions = model.predict_val(X_test)
                val_accuracy = compute_accuracy(val_predictions, y_test)
                val_accuracies[i_nn, i_csvm, idx_fold, k] = val_accuracy
                print("## val accuracy", "k_mers", k_mers, "c_svm", c_svm, "idx_fold", idx_fold, k, "val_accuracy", val_accuracy)
print("val_accuracies", val_accuracies)
sum_val = np.mean(val_accuracies, axis=2) 
std_val = np.std(val_accuracies, axis=2) 
print("sum_val", sum_val)