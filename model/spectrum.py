import numpy as np
from multiprocessing import Pool
from itertools import product
# local import 
from util.config import *


def get_sequences_distributions_spectrum(sequences, all_combinations_letters, k_mers):
    """
    gives one spectrum sequences features
    """
    all_sub_seq = get_all_sub_sequences(sequences, k_mers)
    values = np.zeros([len(all_combinations_letters),])
    for sub_seq in all_sub_seq:
        idx_sub_seq = all_combinations_letters.index(sub_seq)
        values[idx_sub_seq] = values[idx_sub_seq] + 1
    return values


def get_sequences_distributions_one_mismatch(sequences, all_combinations_letters, k_mers):
    '''
    gives one mismatch sequences features
    '''
    all_sub_seq = get_all_sub_sequences(sequences, k_mers)
    values = np.zeros([len(all_combinations_letters),])
    for sub_seq in all_sub_seq:
        idx_sub_seq = all_combinations_letters.index(sub_seq)
        values[idx_sub_seq] = values[idx_sub_seq] + 1
        # allow for one mismatch
        for idx_curr_letter, current_letter in enumerate(sub_seq):
            for letter in LETTERS:
                if letter != current_letter:
                    # copy and modify letter diff from current letter
                    sub_seq_bis = list(sub_seq)
                    sub_seq_bis[idx_curr_letter]= letter
                    mismatch_sub_seq = tuple(sub_seq_bis)
                    idx_sub_seq = all_combinations_letters.index(mismatch_sub_seq)
                    values[idx_sub_seq] = values[idx_sub_seq] + 0.1
    return values


def compute_kernel(X1, X2=None):
    """
    compute gram matrix
    """
    if X2 is None:
        # train
        norm_values = np.empty((len(X1)))
        gram_mat = np.zeros((len(X1), len(X1)))
        for i in range(len(X1)):
            norm_values[i] = np.vdot(X1[i], X1[i])
                        
        for i in range(len(X1)):
            for j in range(i,len(X1)):
                gram_mat[i, j]= np.vdot(X1[i], X1[j])/(norm_values[i] * norm_values[j])**0.5
                gram_mat[j, i] = gram_mat[i, j]
    else:
        # val / test
        gram_mat = np.zeros((len(X1), len(X2)))
        norm_values_X1 = np.empty((len(X1)))
        norm_values_X2 = np.empty((len(X2)))
        for i in range(len(X1)):
            norm_values_X1[i] = np.vdot(X1[i], X1[i])
        for j in range(len(X2)):
            norm_values_X2[j] = np.vdot(X2[j], X2[j])

        for i in range(len(X1)):
            for j in range(len(X2)):
                gram_mat[i, j] = np.vdot(X1[i], X2[j])/(norm_values_X1[i] * norm_values_X2[j])**0.5

    return gram_mat


def get_all_sub_sequences(sequences, k_mers):
    return list(zip(*[sequences[n:] for n in range(k_mers)]))


def get_all_combinations_letters(k_mers):
    """
    return all possibles substrings of length k_mers with voc in LETTERS : A, T, C, G
    """
    return list(product(LETTERS, repeat=k_mers))


def get_repres_spectrum_kernel(X1, k_mers):
    """
    compute the RKHS representation of X1 for spectrum kernel for a substring size of k_mers
    """
    all_combinations_letters = get_all_combinations_letters(k_mers)
    X_test_repres= np.empty([len(X1), len(all_combinations_letters)])
    for i, row in enumerate(X1):
        X_test_repres[i,:] = get_sequences_distributions_spectrum(row, all_combinations_letters, k_mers)
    return X_test_repres


def get_repres_mismatch_kernel(X1, k_mers):
    """
    compute the RKHS representation of X1 for mismatch (one mismatch) kernel for a substring size of k_mers
    """
    all_combinations_letters = get_all_combinations_letters(k_mers)
    X_test_repres= np.empty([len(X1), len(all_combinations_letters)])
    for i, row in enumerate(X1):
        X_test_repres[i,:] = get_sequences_distributions_one_mismatch(row, all_combinations_letters, k_mers)
    return X_test_repres