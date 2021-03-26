import numpy as np
import argparse
from sklearn.model_selection import KFold
# local import
from model.model import Model
from util.config import *
from util.util import *
from model.spectrum import get_repres_spectrum_kernel, get_repres_mismatch_kernel

ks = [0, 1, 2]
train_sequences, train_labels, test_sequence = get_raw_train_data(ks)
# classifier
classifier_params = {}
classifier_type = "svm"
classifier_params["c_svm"] = 0.5 # default value
# kernel 
kernel_name = "mismatch"
kernel_params = {}
kernel_params["load_kernel"] = True
kernel_params["save_kernel"] = True
# hyperparameters 
n_splits = 5
kernel_params["k_mers"] = 6

kfold = KFold(n_splits=n_splits)
preds_list = []
for train_sequence, train_label, test_sequence, k in zip(train_sequences, train_labels, test_sequence, ks):
    print("\n## data set " + str(k))
    bagging_list = []
    print("## compute test train repres")
    test_repres = get_repres_mismatch_kernel(test_sequence, kernel_params["k_mers"])
    train_sequence_repres = get_repres_mismatch_kernel(train_sequence, kernel_params["k_mers"])
    for idx_fold, (train_index, test_index) in enumerate(kfold.split(train_sequence, train_label)):
        print("\n## fold", idx_fold)
        # Params 
        kernel_params["save_name"] = "bagging" + str(idx_fold)
        kernel_params["load_name"] = kernel_params["save_name"]
        # respresentation data in RKHS
        X_train_repres = train_sequence_repres[train_index]
        X_val_repres = train_sequence_repres[test_index]
        # Real data
        X_train, y_train = train_sequence[train_index], train_label[train_index]
        X_val, y_val = train_sequence[test_index], train_label[test_index]
        
        # define model
        model = Model(classifier_type, classifier_params, kernel_name, kernel_params, id_model=str(k))
        # fitting
        model.fit(X_train, y_train, X_train_repres)
        # train accuracy
        train_accuracy = compute_accuracy(model.predict_train(), y_train)
        print("## train accuracy", train_accuracy)
        # val accuracy
        val_predictions = model.predict_val(X_val, X_val_repres, X_train_repres)
        val_accuracy = compute_accuracy(val_predictions, y_val)
        print("## val accuracy", val_accuracy)
        # compute test
        test_predictions = model.predict(test_sequence, test_repres, X_train_repres)
        test_predictions = (test_predictions + 1)/2
        bagging_list.append(test_predictions)
        
    res_bagging = get_poll_result(bagging_list)
    preds_list.append(res_bagging)

write_predictions(preds_list, name="res_bagging")