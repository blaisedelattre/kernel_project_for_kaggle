import numpy as np
from itertools import combinations
from model.substring import word2ngrams
from util.config import ALPHABET


def combination(str_):
    '''
    All the way to choose a substring inside a string
    '''
    comb = [''.join(l) for i in range(len(str_)) for l in combinations(str_, i+1)]
    return comb


def n_tuple(str_,n):
    '''
    All the way to choose a sub-string of size n inside a string
    '''
    ntuple = [el for el in combination(str_) if len(el)==n]
    if n==2:
        for letter in str_:
            ntuple.append(letter + letter)
    return ntuple


def frequency(n,str_):
    '''
    All the frequencies of the sub-strings of size n among all the words of letters coming from ALPHABET
    '''
    ntuple = n_tuple(ALPHABET,n)
    all_freq = {key: 0 for (key) in ntuple}
    str_k = word2ngrams(str_,n)
    for i in str_k:
        if i in all_freq:
            all_freq[i] += 1
        else:
            all_freq[i] = 1
    sum_p = sum(all_freq.values())
    for i in all_freq:
        all_freq[i] = float(all_freq[i]/sum_p)
    return all_freq,len(str_k)


def all_frequency(list_of_seq):
    '''
    The same as the function above but for a list of strings
    '''
    dic = {str_: 0 for str_ in ALPHABET}
    for seq in list_of_seq:
        d = frequency(1,seq)[0]
        for key in d:
            dic[key]+=d[key]
    return dic


def compute_substitution_matrix(X,substitution_matrix = np.zeros((4,4))):
    '''
    Calculation of the substitution matrix from the list of sequence X in the same way as the BLOSUM62 calculation
    '''
    X_col =  np.array([list(s) for s in X]).T.tolist()
    score_pairwise = []
    length = []
    for col in X_col:
        freq, len = frequency(2,col)
        score_pairwise.append(freq)
        length.append(len)
    list_pairwise = n_tuple(ALPHABET,2)

    total = sum(length)
    all_freq = all_frequency(X)
    for pairwise in list_pairwise:
        total_score = sum([score[pairwise] for score in score_pairwise])/total
        letter_0 = pairwise[0]
        letter_1 = pairwise[1]
        i = ALPHABET.index(letter_0)
        j = ALPHABET.index(letter_1)
        if i==j:
            substitution_matrix[i,j] = 10000*np.log2(total_score/(all_freq[letter_0]**2))
        else:
            substitution_matrix[i,j] = 10000*np.log2(total_score/(2*all_freq[letter_0]*all_freq[letter_1]))
    return substitution_matrix


def LA_kernel(substitution_matrix,beta,d,e,x,y):
    '''
    Dynamic programming step to calculate the LA kernel between x and y 
    '''
    n = len(x)
    M = np.zeros((n, n))
    X = np.zeros((n,n))
    Y = np.zeros((n,n))
    X2 = np.zeros((n,n))
    Y2 =  np.zeros((n,n))
    for i in range(1,n):
        for j in range(1,n):
            index_i = ALPHABET.index(x[i-1])
            index_j = ALPHABET.index(y[j-1])
            M[i,j] = np.exp(beta*(substitution_matrix[index_i,index_j]))*(1+X[i-1,j-1]+Y[i-1,j-1] +
            M[i-1,j-1])
            X[i,j] =  np.exp(beta*d)*M[i-1,j] + np.exp(beta*e)*X[i-1,j]
            Y[i,j] =  np.exp(beta*d)*(M[i,j-1]+X[i,j-1])+ np.exp(beta*e)*Y[i,j-1]
            X2[i,j] = M[i-1,j] + X2[i-1,j]
            Y2[i,j] = M[i,j-1] + X2[i,j-1] + Y2[i,j-1]
    return (1 + X2[-1,-1]+Y2[-1,-1]+M[-1,-1])


if __name__ == "__main__":
    print('Successfully passed.')

    ######### TEST #########

    X1 = 'AAGGCCGAGCCCGGCGCGGACGCAGGCGGCTCCGGGCGGGCTCAGCACCCCCAGGCACCGTCTCCTAGTGACCGCGGCGCTCGCGGGCCTGGCGGCCGTTG'
    X2 = 'TCTGGGCTCTTAATGTAAAGGTTGCCACTGATGCTGTGTCACCAGCGCCCCCTCTGTGCATCCTTAGGAGCTGCGGGGGCCAGGAGGGAGGGGGAGGCGCG'
    X = [X1,X2]
    substitution_matrix = compute_substitution_matrix(X)

    #print(substring_kernel(X1,X2,0.5,3))
    print(LA_kernel(substitution_matrix,beta=0.05,d=1,e=11,x=X1,y=X2))
