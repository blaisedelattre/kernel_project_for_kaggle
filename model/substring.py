import itertools

def word2ngrams(text, n=3, exact=True):
    '''
    Convert text into character ngrams.
    '''
    return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]


def sublist2word(text, sublist, length=False):
    '''
    Allows you to obtain the substring from the corresponding indices in the starting string.
    '''
    str = ''
    for index in sublist:
        str = str + text[index]
    #return (str, sublist[-1]-sublist[0]+1)
    return str


def allSublist(k, n):
    '''
    All subsets of size k of {1,...,n}.
    '''
    return list(itertools.combinations(set([k for k in range(1,n)]), k))


def allSubstring(text):
    '''
    Returns all substrings of a text.
    '''
    res = [test_str[i: j] for i in range(len(test_str))
          for j in range(i + 1, len(test_str) + 1)]
    return res


def substring(text, k=3):
    '''
    Returns all substrings of a text for a certain size k.
    '''
    return [word for word in allSubstring(text) if len(word) == size]


def checkEqual(text, sublist, ngram):
    '''
    To check that a sublist matches an n-gram.
    '''
    return sublist2word(text,sublist)==ngram


def naive_mapping(text, ngram, lambda_, k):
    '''
    Naively calculates the mapping as in the course for the Substring kernel.
    '''
    n = len(text)
    sublist_of_index = allSublist(k, n)
    sum = 0
    for sublist in sublist_of_index:
        if checkEqual(text, sublist, ngram):
            sum = sum + lambda_**(sublist[-1]-sublist[0]+1)
    return sum


def substring_kernel(X1, X2, lambda_, k):
    '''
    Compute naively the substring kernel by enumerating all of the substring.
    '''
    ngrams = word2ngrams('ACGT', n=k, exact=True)
    sum = 0
    for ngram in ngrams:
        sum = sum + naive_mapping(X1, ngram, lambda_, k) * naive_mapping(X2, ngram, lambda_, k)
    return sum


def get_memo(X1,X2,lambda_,k):
    memo = {}
    def Bk_function(X1,X2,lambda_,k):
        if not k in memo:
            if k==0:
                memo[k] = 1
            elif min(len(X1),len(X2))<k:
                memo[k] = 0
            else:
                memo[k] = (lambda_*Bk_function(X1[:-1],X2,lambda_,k) + lambda_*Bk_function(X1,X2[:-1],lambda_, k)
                        + lambda_**2*(-Bk_function(X1[:-1],X2[:-1],lambda_,k) + (X1[-1]==X2[-1])*Bk_function(X1[:-1],X2[:-1],lambda_,k-1)))
        return memo[k]

    Bk_function(X1,X2,lambda_,k)
    return memo[k]


def equal_last_element(text,a):
    n = len(text)
    list_ = [pos for pos, char in enumerate(text) if char == a]
    return list_


def substring_rec(X1,X2,lambda_,k):
    memo = {}
    def Kk_function(X1,X2,lambda_,k):
        if not k in memo:
            if k==0:
                memo[k] = 1
            elif min(len(X1),len(X2))<k:
                memo[k] = 0
            else:
                index_ = equal_last_element(X2,X1[-1])
                sum  = 0
                for index in index_:
                    sum = sum + get_memo(X1[:-1],X2[:index],lambda_,k-1)
                memo[k] = Kk_function(X1[:-1],X2,lambda_,k) + lambda_**2*sum
        return memo[k]

    Kk_function(X1,X2,lambda_,k)
    return memo[k]


if __name__ == "__main__":
    print('Successfully passed.')

    ######### TEST #########

    X1 = 'AAGGCCGAGCCCGGCGCGGACGCAGGCGGCTCCGGGCGGGCTCAGCACCCCCAGGCACCGTCTCCTAGTGACCGCGGCGCTCGCGGGCCTGGCGGCCGTTG'
    X2 = 'TCTGGGCTCTTAATGTAAAGGTTGCCACTGATGCTGTGTCACCAGCGCCCCCTCTGTGCATCCTTAGGAGCTGCGGGGGCCAGGAGGGAGGGGGAGGCGCG'
    #print(substring_kernel(X1,X2,0.5,3))
    print(substring_rec(X1,X2,0.5,3))
