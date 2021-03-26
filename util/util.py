import pandas as pd
from os.path import join, exists
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons
from sklearn.utils import shuffle
import datetime
from sklearn.model_selection import train_test_split
import pickle
# local import
from util.config import *


def convert_letter_sequence_to_number_sequence(sequences):
    # what format A = (1, 0, 0 ,0) B = (0, 1, 0, 0) or ?
    pass

def get_train_features_name(k):
    return TRAINING_SEQUENCE + str(k) +  "_mat100" + EXTENSION_NAME

def get_test_features_name(k):
    return TEST_SEQUENCE + str(k) + "_mat100" + EXTENSION_NAME

def get_train_sequence_name(k):
    return TRAINING_SEQUENCE + str(k) + EXTENSION_NAME

def get_test_sequence_name(k):
    return TEST_SEQUENCE + str(k) + EXTENSION_NAME

def get_train_labels_name(k):
    return TRAINING_LABEL + str(k) + EXTENSION_NAME

def get_raw_train_data(ks = Ks):
    """
    return train sequences and labels
    remember we have to predict separateky each data set separately
    """
    train_sequences = []
    train_labels = []
    test_sequences = []
    for k in ks:
        train_sequences.append(pd.read_csv(join(DATA_FILE, get_train_sequence_name(k)))["seq"])
        y = pd.read_csv(join(DATA_FILE, get_train_labels_name(k)))["Bound"].to_numpy()
        
        #put index from 0 1 to -1 1
        y = 2*y -1
        train_labels.append(y)
        
        test_sequences.append(pd.read_csv(join(DATA_FILE, get_test_sequence_name(k)))["seq"])
    return train_sequences, train_labels, test_sequences


def do_data_auguementation(X, y):
    """
    --X dataset with sequences of letters
    --y label 1 or 0
    for letter sequence double the dataset by taking complementary of sequences
    A -> T, T-> A, G -> C, C -> G
    """
    extra_X = X.copy()
    extra_y = y.copy()
    trans_table = str.maketrans({'A': 'T', 'T': 'A', 'C':'G', 'G':'C'})
    for row in extra_X:
        row = row.translate(trans_table)
    return np.concatenate((X, extra_X), axis=0), np.concatenate((y, extra_y), axis=0)

def get_features_train_data(ks=Ks, normalized=False, do_pca=False, do_tsne=False):
    """
    return train features and labels
    remember we have to predict separateky each data set separately
    """
    print("## retrieve data")
    train_sequences = []
    train_labels = []
    test_sequence = []
    for k in ks:
        X = pd.read_csv(join(DATA_FILE, get_train_features_name(k)),index_col=False, header=None, sep=" ").to_numpy()
        X_test = pd.read_csv(join(DATA_FILE, get_test_features_name(k)),index_col=False, header=None, sep=" ").to_numpy()
        y = pd.read_csv(join(DATA_FILE, get_train_labels_name(k)), index_col=0).to_numpy()
        # remove index col
        y = y[:, -1]
        #put index from 0 1 to -1 1
        y = 2*y -1
        # normalize 
        if normalized:
            X = standardize_data(X)
            X_test = standardize_data(X_test)
        if do_pca:
            pca_50 = PCA(n_components=50)
            X = pca_50.fit_transform(X)
            X_test = pca_50.fit_transform(X_test)
        if do_tsne:
            tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
            X = tsne.fit_transform(X)
            X_test = tsne.fit_transform(X_test)
        train_sequences.append(X)
        test_sequence.append(X_test)
        train_labels.append(y)
    return train_sequences, train_labels, test_sequence



def create_training_and_validation_set(sequences, labels, percentage_val=0.20, data_aug_train=False, data_aug_all=False, random_state=42):
    """
    create training and validation set and apply data auguementation or not
    """
    # shuffle
    sequences, labels = shuffle(sequences, labels)

    if data_aug_all:
        print("## data aug all data")
        sequences, labels = do_data_auguementation(sequences, labels)

    if percentage_val > 0:
        train_sequence, val_seq, train_label, val_lablel = train_test_split(sequences, labels, test_size=percentage_val, random_state=random_state)
    else: # pas de val set 
        val_seq, val_lablel = [], []
        train_sequence, train_label = sequences, labels
    if data_aug_train:
        print("## data aug train data")
        train_sequence, train_label = do_data_auguementation(train_sequence, train_label)

    return train_sequence, train_label, val_seq, val_lablel


def get_dummy_train_data(n=100, d=100, normalized=False, do_pca=False, do_tsne=False):
    """
    return train features and labels
    """
    print("## retrieve dummy data")
    X, y = gen_dummy_data()
    n, d = X.shape
    # normalize 
    if normalized:
        X = standardize_data(X)
    if do_pca:
        pca_50 = PCA(n_components=min(50, d, n))
        X = pca_50.fit_transform(X)
    if do_tsne:
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        X = tsne.fit_transform(X)
    return [X], [y]


def gen_dummy_data():
    """
    """
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 1000)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 1000)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    X_train = np.vstack((X1, X2))
    y_train = np.hstack((y1, y2))
    return X_train, y_train


def compute_accuracy(preds, labels):
    """
    accuracy  = VP /(VP + FP)
    """
    return np.mean(preds == labels) * 100


def standardize_data(arr):
    """
    normalize data
    """    
    rows, columns = arr.shape
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    for column in range(columns):
        mean = np.mean(arr[:,column])
        std = np.std(arr[:,column])
        tempArray = np.empty(0)
        for element in arr[:,column]:
            tempArray = np.append(tempArray, ((element - mean) / std))
        standardizedArray[:,column] = tempArray
    return standardizedArray


def get_poll_result(bagging_list):
    """
    compute poll for bagging method
    """
    res = np.mean(np.array(bagging_list), axis=0)
    res[res >= 1/2] = 1
    res[res < 1/2] = 0
    return res


def write_predictions(preds, args=None, val_accuracies=[1], name=""):
    """
    write prediction in csv file
    """
    if args is not  None:
        name = str(args.kernel) + "_" + args.cls + "_" 
    preds = np.concatenate(preds)
    y_test = pd.DataFrame({'Id': np.arange(len(preds)), 'Bound': preds})
    y_test.Bound = y_test.Bound.replace(-1, 0).astype(int)
    tag = name + "_".join(map(str, map(round, val_accuracies)))
    y_test.to_csv('y_test_' + str(tag) + EXTENSION_NAME, index=False)


def create_dir_(dir_name):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def dump_object(kernel_mat, name, save=False):
    if save:
        pickle.dump(kernel_mat, open(join(SAVE_DATA_FILE, name), "wb" ))

def load_object(name):
    if exists(join(SAVE_DATA_FILE, name)):
        m = pickle.load(open(join(SAVE_DATA_FILE, name), "rb"))
    else:
        m = None
    return m