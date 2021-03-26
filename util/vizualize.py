import matplotlib.pyplot as plt
import numpy as np; np.random.seed(42)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import time
import pandas as pd
#local
from util.util import get_features_train_data, standardize_data



def visualize_data():
    train_sequences, train_labels = get_features_train_data()
    do_PCA(train_sequences[0], train_labels[0])

def do_PCA(X, y):
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    rndperm = np.random.permutation(df.shape[0])
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df.loc[rndperm,:],
        legend="full",
        alpha=0.3
    )
    plt.show()

    data_subset = df[feat_cols].values
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3
    )

    plt.show()

    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df['tsne-pca50-one'] = tsne_pca_results[:,0]
    df['tsne-pca50-two'] = tsne_pca_results[:,1]
    sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.3,
    )
    plt.title("pca 50 + t SNE")
    plt.show()

def do_SVE(X, y):
    X = train_sequences[0]
    X = standardize_data(X)
    U, s, V = np.linalg.svd(X)
    Ar = X.dot( V.transpose() )
    plt.plot(Ar[:,0], Ar[:,1], '.')
    plt.axis('equal')
    plt.show()



