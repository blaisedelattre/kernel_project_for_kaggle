import numpy as np
import argparse
# local import
from model.model import Model
from util.config import *
from util.util import *
from util.vizualize import visualize_data


def parse():
    parser = argparse.ArgumentParser()
    # basic options
    parser.add_argument("--data", type=str, default = "features", help="launch classifier with features representation, dummy")
    parser.add_argument("--cls", type=str, default = "svm", help="use svm, l_regression")
    parser.add_argument("--kernel", type=str, default = "rbf", help="define kernel to use rbf, polynomial, linear etc")
    parser.add_argument("--mode", type=str, default = "train", help="do training, testing, visual")
    parser.add_argument("--write_pred", action='store_true', help="write predictions")
    parser.add_argument("--percen_val", type=float, default = 0.2, help="percentage of val set")
    
    # options to save load kernel
    parser.add_argument("--save_kernel", action='store_true', help="save kernel matrix")
    parser.add_argument("--save_name", type=str, default = "last_kernel", help="save kernel name default last_kernel.pkl")
    parser.add_argument("--load_kernel", action='store_true', help="load kernel or not")
    parser.add_argument("--load_name", type=str, default = "last_kernel", help="load kernel default is last_kernel.pkl")
    
    # data auguementation
    parser.add_argument("--data_aug_train", action='store_true', help="do data auguementation on train")
    parser.add_argument("--data_aug_all", action='store_true', help="do data auguementation on train and val")
    
    # svm params
    parser.add_argument("--c_svm", type=float, default = 1, help="C parameter for svm")

    # logistic params
    parser.add_argument("--lbda", type=float, default = 0.01, help="lambda parameter for logistic")

    # normalize data
    parser.add_argument("--normalized", action='store_true', help="normalize data")
    parser.add_argument("--do_pca", action='store_true', help="take pca representation of data")
    parser.add_argument("--do_tsne", action='store_true', help="take tsne representation of data")
    
    # rbf kernel
    parser.add_argument("--sigma", type=float, default = 0, help="sigma parameter for rbf, by default set with formula")
    # polynomial kernel
    parser.add_argument("--offset", type=float, default = 1, help="offset for polynomial kernel")
    parser.add_argument("--dim", type=float, default = 100, help="dim for polynomial kernel")
    # spectrum kernel
    parser.add_argument("--k_mers", type=int, default =5, help="k_mers hyperpaamter for spectrum/mismatch kernel")

    args = parser.parse_args()
    print(args)
    return args


def train(args):
    # define wished datasets
    Cs = [1, 0.9, 0.9] # best c svm param for datasets TODO hyperparameters algo 
    ks = [1]
    if args.write_pred:
        ks = Ks

    # retrieve data
    if args.data == "features":
        train_sequences, train_labels, test_sequence = get_features_train_data(ks, normalized=args.normalized, do_pca=args.do_pca, do_tsne=args.do_tsne)
    elif args.data == "dummy":
        train_sequences, train_labels = get_dummy_train_data(n=1000, d=100, normalized=args.normalized, do_pca=args.do_pca, do_tsne=args.do_tsne)
    elif args.data == "raw":
        train_sequences, train_labels, test_sequence = get_raw_train_data(ks)

    # is data augu posssible
    data_aug_train = args.data_aug_train and args.data == "raw" 
    data_aug_all = args.data_aug_all and args.data == "raw" 
    # do we have val set
    is_val_set = args.percen_val != 0

    # deal with classifier params
    classifier_type = args.cls
    classifier_params = {}
    if classifier_type == "svm":
        classifier_params["c_svm"] = args.c_svm
        
    if classifier_type == "l_regresison":
        classifier_params["lbda"] = args.lbda

    # deal with kernel params
    kernel_name = args.kernel
    kernel_params = {}
    kernel_params["load_kernel"] = args.load_kernel
    kernel_params["load_name"] = args.load_name
    kernel_params["save_kernel"] = args.save_kernel
    kernel_params["save_name"] = args.save_name
    if kernel_name in ("rbf", "sum"):
        kernel_params["sigma"] = args.sigma
    if kernel_name in ("poly", "sum"):
        kernel_params["poly_offset"] = args.offset
        kernel_params["dim"] = args.dim
    if kernel_name in ("spectrum", "mismatch"):
        kernel_params["k_mers"] = args.k_mers

    # launch training
    preds_list = []
    val_accuracies = []
    for train_sequence, train_label, test_sequence, k in zip(train_sequences, train_labels, test_sequence, ks):
        print("\n## data set " + str(k))
        # define train and val set
        train_sequence, train_label, val_seq, val_label = create_training_and_validation_set(train_sequence, train_label, 
                                                                                            percentage_val=args.percen_val,
                                                                                            data_aug_all=data_aug_all,
                                                                                            data_aug_train=data_aug_train)
        # define model
        model = Model(classifier_type, classifier_params, kernel_name, kernel_params, id_model=str(k))
        # fitting
        model.fit(train_sequence, train_label)
        # train accuracy
        train_accuracy = compute_accuracy(model.predict_train(), train_label)
        print("## train accuracy", train_accuracy)
        # val accuracy
        if is_val_set:
            val_predictions = model.predict_val(val_seq)
            val_accuracy = compute_accuracy(val_predictions, val_label)
            val_accuracies.append(val_accuracy)
            print("## val accuracy", val_accuracy)
        # test
        if args.write_pred:
            test_predictions = model.predict(test_sequence)
            preds_list.append(test_predictions)

    # write predictions
    if args.write_pred:
        write_predictions(preds_list, args, val_accuracies)

   
def main():
    args = parse()
    # create folder
    create_dir_(SAVE_DATA_FILE)
    if args.mode == "train":
        train(args)
    elif args.mode == "visual":
        visualize_data()
    

main()