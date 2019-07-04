import warnings
import glob

import numpy as np
import matplotlib.pyplot as plt

import torch

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import ParameterGrid
from scipy import sparse
from tqdm import tqdm

# GPU or CPU t-SNE implementation
if torch.cuda.is_available():
    from tsnecuda import TSNE
else:
    from sklearn.manifold import TSNE



########################################################################################################################
# DATA

features_path = '../data/features/resnet.npy'
icon_features = np.load(features_path)[:100000]

print('got data \n')

########################################################################################################################
# DIMENSIONALITY REDUCTION

"""
First of all dimensionality has to be reduced to ~ 50 features and only then t-SNE can be used due to computational 
limitations. Beforehand TruncatedSVD will be used to reduce dimensions, it can efficiently be used with sparse data as 
well. Different hyper-parameters will be tested for t-SNE.
"""

# reduce dimensions with TruncatedSVD
icon_features_reduced = TruncatedSVD(n_components=50).fit_transform(sparse.csr_matrix(icon_features))

print('data reduced \n')

# t-SNE
tsne_path = '../data/tsne'
tsne_path_plot = './plots/tsne'
parameters = {'perplexity': [20, 35, 50], 'learning_rate': [50, 125, 200], 'init': ['random', 'pca']}

for param in tqdm(ParameterGrid(parameters), desc='t-SNE hyper-parameter search'):

    # embedding
    embedded_icon_features = TSNE(n_components=2, **param).fit_transform(icon_features_reduced)

    # save numpy array
    features_embedded_path = tsne_path + '/{}_{}_{}.npy'.format(*param.values())
    np.save(features_embedded_path, embedded_icon_features)

    # plot data
    plt.figure(figsize=(6, 6))
    plt.title('t-SNE parameters - \n init: {}, learning rate: {}, perplexity: {}'.format(*param.values()),
              size=12, weight='bold')
    plt.scatter(x=embedded_icon_features[:, 0], y=embedded_icon_features[:, 1], alpha=0.1)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(tsne_path_plot+'/{}_{}_{}.png'.format(*param.values()))
    plt.clf()

print('t-SNE finished \n')


########################################################################################################################
# CLUSTERING

# helper function
def clustering(_data, _path, _param):

    # initialize figure
    plt.figure(figsize=(15, 3))
    plt.tight_layout()
    plt.subplots_adjust(top=0.7)

    plt.suptitle('t-SNE parameters - init: {}, learning rate: {}, perplexity: {}'.format(*_param),
                 size=14, weight='bold')

    plot_num = 1

    params = {'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': 2,
              'min_samples': 20,
              'min_cluster_size': 0.1}

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(_data)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # Create cluster objects
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN()
    optics = cluster.OPTICS(min_samples=params['min_samples'], min_cluster_size=params['min_cluster_size'])
    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",
                                                      n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('Mini Batch \n KMeans', two_means),
        ('Affinity \n Propagation', affinity_propagation),
        ('Mean Shift', ms),
        ('Spectral \n Clustering', spectral),
        ('Ward', ward),
        ('Agglomerative \n Clustering', average_linkage),
        ('DBSCAN', dbscan),
        ('OPTICS', optics),
        ('Birch', birch),
        ('Gaussian \n Mixture', gmm)
    )

    for name, algorithm in tqdm(clustering_algorithms, desc='clustering', leave=False):
        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " + "connectivity matrix is [0-9]{1,2}" +
                        " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" + " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        # predict labels
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # plot
        plt.subplot(1, len(clustering_algorithms), plot_num)
        plt.title(name, size=11)

        colors = np.array(list(islice(cycle(['red', 'blue', 'green', 'black']), int(max(y_pred) + 1))))

        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred], alpha=0.1)

        plt.xticks(())
        plt.yticks(())

        plot_num += 1

    plt.savefig(_path)
    plt.close('all')


# clustering for every embedded data set
for p in tqdm(glob.glob(tsne_path + '/*.npy'), desc='data sets'):
    # get parameter previously used for t-SNE
    split = p.split('_')
    init = split[0].split('/')[-1]
    learning_rate = split[1]
    perplexity = split[2][:-4]

    parameters = [init, learning_rate, perplexity]

    # data
    data = np.load(p)

    # path to save clustering plot
    path = tsne_path_plot + '/'
    path = path + (p.split('/')[-1][:-4]+'_clustering.png')

    # actual clustering
    clustering(_data=data, _path=path, _param=parameters)

print('clustering done \n')