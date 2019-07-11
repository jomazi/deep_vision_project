import warnings
import glob
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from scipy.spatial.distance import pdist, squareform

from fcts.load_data import load_data

warnings.filterwarnings("ignore")

########################################################################################################################
# CONFIG

parser = argparse.ArgumentParser(description='Clustering of ResNet features')

parser.add_argument('--low', action='store_false', help='Cluster low dimensional data (alternative: False '
                                                        '--> high dimensional data)? default=True')
parser.add_argument('--vae', action='store_true', help='Cluster latent space of trained VAE (only possible if'
                                                       'low=True)? default=False')
parser.add_argument('--categories', type=int, default=5, help='Number of clusters (should not exceed 5); default=5')
parser.add_argument('--num_images', type=int, default=10000, help='Number of images to use; default=10000')

opt = parser.parse_args()

# determine directory of this script
cur_dir = os.path.dirname(os.path.abspath(__file__))

# path to features
features_path = os.path.join(cur_dir, '../data/features/resnet.npy')
features_path_vae = os.path.join(cur_dir, '../data/features/dcgan32.npy')

# path to save labels
labels_path = os.path.join(cur_dir, '../data/labels/')

# path to save plots
ppath = os.path.join(cur_dir, './plots/categories/')

########################################################################################################################
# DATA

# check if features exist
assert os.path.isfile(features_path), 'Get ResNet features first!'
assert os.path.isfile(features_path_vae), 'Get VAE features first!'

icon_features = np.load(features_path)[:opt.num_images]
icon_features_vae = np.load(features_path_vae)[:opt.num_images]

print('got data \n')

########################################################################################################################
# DIMENSIONALITY REDUCTION

"""
If low dimensional data has to be clustered, dimensionality has to be reduced to ~ 50 features first and then t-SNE 
is used due to computational limitations to further reduce dimensions. Beforehand, a PCA will be applied.
"""

if opt.low:
    # load VAE features if desired
    if opt.vae:
        icon_features_reduced = icon_features_vae
    else:
        # reduce dimensions with TruncatedSVD
        icon_features_reduced = PCA(n_components=50).fit_transform(icon_features)

    print('data reduced \n')

    # t-SNE
    if opt.vae:
        tsne_path = '../data/tsne/vae'
        tsne_path_plot = './plots/tsne/vae'
    else:
        tsne_path = '../data/tsne'
        tsne_path_plot = './plots/tsne'
    parameters = {'perplexity': [20, 35, 50], 'learning_rate': [50, 125, 200]}

    metrics = {'euclidean': squareform(pdist(icon_features_reduced), 'euclidean'),
               'cityblock': squareform(pdist(icon_features_reduced), 'cityblock'),
               'cosine': squareform(pdist(icon_features_reduced), 'cosine'),
               'correlation': squareform(pdist(icon_features_reduced), 'correlation'),
               'Lk12': squareform(pdist(icon_features_reduced, lambda u, v: (np.abs((u-v))**(1/2)).sum()**2)),
               'Lk13': squareform(pdist(icon_features_reduced, lambda u, v: (np.abs((u-v))**(1/3)).sum()**3))}

    for metric, X in tqdm(metrics.items(), desc='metrics'):
        for param in tqdm(ParameterGrid(parameters), desc='t-SNE hyper-parameter search'):

            # embedding
            embedded_icon_features = TSNE(n_components=2, metric='precomputed', **param).fit_transform(X)

            # save numpy array
            features_embedded_path = tsne_path + '/{}_{}_{}.npy'.format(metric, *param.values())
            np.save(features_embedded_path, embedded_icon_features)

            # plot data
            plt.figure(figsize=(6, 6))
            plt.title('t-SNE parameters - \n metric: {}, learning rate: {}, perplexity: {}'.format(metric, *param.values()),
                      size=12, weight='bold')
            plt.scatter(x=embedded_icon_features[:, 0], y=embedded_icon_features[:, 1], alpha=0.1)
            plt.xticks(())
            plt.yticks(())
            plt.savefig(tsne_path_plot+'/{}_{}_{}.png'.format(metric, *param.values()))
            plt.close('all')

    print('dimensionality reduction finished \n')


########################################################################################################################
# CLUSTERING

# helper function to visualize logos of different categories/clusters
def vis_cat(_path, images, categories, labels):

    """
    :param _path: path to save image; str
    :param images: images to plot; numpy array
    :param categories: categories; list[int]
    :param labels: labels of passed images
    """

    # initialize figure
    plt.figure(figsize=(12, 8))
    plt.tight_layout()

    plt.suptitle('Logos of different categories assigned by clustering', size=14, weight='bold')

    plot_num = 1

    print('categories: ', categories)

    for i, c in enumerate(categories):
        print('category: {}'.format(c))

        # data
        img_c = images[labels == c]
        print('number of images: {}'.format(len(img_c)))

        for j in range(10):
            plt.subplot(len(categories), 10, plot_num)

            # add title for img in center
            if j == 0:
                plt.title('Category: {}'.format(c+1), size=11)

            # plot if image available
            try:
                plt.imshow(img_c[j])
                plt.xticks(())
                plt.yticks(())
            except Exception as e:
                plt.axis('off')

            plot_num += 1

    # save
    plt.savefig(_path)
    # clean up
    plt.close('all')
    plt.clf()


# helper function
def clustering(_data, path_image, _path_labels, _param, plot):

    """
    :param _data: data that will be clustered; numpy array
    :param path_image: path to save image (only necessary if plot=True); str
    :param _path_labels: path to save labels; str
    :param _param: parameter used for dimensionality reduction (only necessary if plot=True); dict
    :param plot: plot clustered data (only possible for 2d); bool
    """

    if plot:
        # initialize figure
        plt.figure(figsize=(15, 3))
        plt.tight_layout()
        plt.subplots_adjust(top=0.7)

        plt.suptitle('t-SNE parameters - metric: {}, learning rate: {}, perplexity: {}'.format(*_param),
                     size=14, weight='bold')

        plot_num = 1

    params = {'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': opt.categories,
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
    average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",
                                                      n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = {
        'Mini Batch \n KMeans': two_means,
        'Mean Shift': ms,
        'Spectral \n Clustering': spectral,
        'Ward': ward,
        'Agglomerative \n Clustering': average_linkage,
        'DBSCAN': dbscan,
        'OPTICS': optics,
        'Birch': birch,
        'Gaussian \n Mixture': gmm
    }

    # names of algorithms that are compatible to use within paths
    algo_names_compatible = ['Mini_Batch_KMeans', 'Mean_Shift', 'Spectral_Clustering', 'Ward',
                             'Agglomerative_Clustering', 'DBSCAN', 'OPTICS', 'Birch', 'Gaussian_Mixture']

    labels = []

    for name, algorithm in tqdm(clustering_algorithms.items(), desc='clustering', leave=False):

        # fit
        algorithm.fit(X)

        # predict labels
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        # store labels
        labels.append(y_pred)

        if plot:
            # plot
            plt.subplot(1, len(clustering_algorithms), plot_num)
            plt.title(name, size=11)

            colors = np.array(list(islice(cycle(['red', 'blue', 'green', 'yellow', 'magenta', 'black']),
                                          int(max(y_pred) + 1))))

            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred], alpha=0.1)

            plt.xticks(())
            plt.yticks(())

            plot_num += 1

    if plot:
        plt.savefig(path_image)
        plt.close('all')

    # save labels
    np.save(_path_labels, np.array(labels))

    # visualize categories

    # labels
    labels = np.load(_path_labels)
    num_img = np.shape(labels)[-1]
    # data
    imgs = load_data()[:num_img]

    for i, name in enumerate(algo_names_compatible):
        _p = ppath + _path_labels.split('/')[-1][:-10]
        _p = _p + '{}_categories.png'.format(name)

        # actual visualization
        vis_cat(_p, imgs, np.unique(labels[i]), labels[i])


if opt.low:
    # clustering for every embedded data set
    for p in tqdm(glob.glob(tsne_path + '/*.npy'), desc='data sets'):
        # get parameter previously used for t-SNE
        split = p.split('_')
        metric = split[0].split('/')[-1]
        learning_rate = split[1]
        perplexity = split[2][:-4]

        parameters = [metric, learning_rate, perplexity]

        # data
        data = np.load(p)

        # path to save clustering plot
        path = tsne_path_plot + '/'
        path = path + (p.split('/')[-1][:-4]+'_clustering.png')
        path_labels = labels_path + (p.split('/')[-1][:-4]+'_labels.npy')

        # actual clustering
        clustering(_data=data, path_image=path, _path_labels=path_labels, _param=parameters, plot=True)

else:
    # actual clustering
    path_labels = labels_path + 'resnet_features_labels.npy'
    clustering(_data=icon_features, path_image='-', _path_labels=path_labels, _param=dict(), plot=False)


print('clustering done \n')

