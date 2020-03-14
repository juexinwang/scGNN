import numpy as np
import sklearn

def calculate_mmd(k1, k2, k12):
    """ Calculates MMD given kernels for batch1, batch2, and between batches """
    return k1.sum()/(k1.shape[0]*k1.shape[1]) + k2.sum()/(k2.shape[0]*k2.shape[1]) - 2*k12.sum()/(k12.shape[0]*k12.shape[1])

def get_cluster_merging(embedding, clusters):
        if len(np.unique(clusters))==1: return clusters

        clusters = clusters - clusters.min()
        clusts_to_use = np.unique(clusters)
        mmdclusts = np.zeros((len(clusts_to_use), len(clusts_to_use)))
        for i1, clust1 in enumerate(clusts_to_use):
            for i2, clust2 in enumerate(clusts_to_use[i1 + 1:]):
                ei = embedding[clusters == clust1]
                ej = embedding[clusters == clust2]
                ri = list(range(ei.shape[0])); np.random.shuffle(ri); ri = ri[:1000];
                rj = list(range(ej.shape[0])); np.random.shuffle(rj); rj = rj[:1000];
                ei = ei[ri, :]
                ej = ej[rj, :]

                k1 = sklearn.metrics.pairwise.pairwise_distances(ei, ei)
                k2 = sklearn.metrics.pairwise.pairwise_distances(ej, ej)
                k12 = sklearn.metrics.pairwise.pairwise_distances(ei, ej)

                mmd = 0
                for sigma in [.01, .1, 1., 10.]:
                    k1_ = np.exp(- k1 / (sigma**2))
                    k2_ = np.exp(- k2 / (sigma**2))
                    k12_ = np.exp(- k12 / (sigma**2))

                    mmd += calculate_mmd(k1_, k2_, k12_)
                mmdclusts[i1, i1 + i2 + 1] = mmd
                mmdclusts[i1 + i2 + 1, i1] = mmd

        clust_to = {}
        for i1 in range(mmdclusts.shape[0]):
            for i2 in range(mmdclusts.shape[1]):
                argmin1 = np.argsort(mmdclusts[i1, :])[1]
                argmin2 = np.argsort(mmdclusts[i2, :])[1]
                if argmin1 == (i1 + i2) and argmin2 == i1 and i2 > i1:
                    clust_to[i2] = i1


        for c in clust_to:
            mask = clusters == c
            clusters[mask.tolist()] = clust_to[c]

        clusts_to_use_map = [c for c in clusts_to_use.tolist() if c not in clust_to]
        clusts_to_use_map = {c:i for i,c in enumerate(clusts_to_use_map)}

        for c in clusts_to_use_map:
            mask = clusters==c
            clusters[mask.tolist()] = clusts_to_use_map[c]


        return clusters

#TODO
def get_clusters(embedding, binmin=100, max_clusters=1000, verbose=True):
    """
    Get cluster assignments from the ID regularization layer.
    :param load: the loader object to iterate over
    :param binmin: points in a cluster of less than this many points will be assigned the unclustered "-1" label
    :param max_clusters: going through the clusters can take a long time, so optionally abort any attempt to go
                            through more than a certain number of clusters
    :param verbose: whether or not to print the results of the clustering
    """
    acts = self.get_layer(load, 'layer_c')
    if isinstance(acts, list) or isinstance(acts, tuple):
        acts = acts[0]

    acts = acts / acts.max()
    binarized = np.where(acts > .000001, 1, 0)

    unique_rows, counts = np.unique(binarized, axis=0, return_counts=True)
    unique_rows = unique_rows[counts > binmin]

    num_clusters = unique_rows.shape[0]
    if num_clusters > max_clusters:
        print("Too many clusters ({}) to go through...".format(num_clusters))
        return num_clusters, np.zeros(acts.shape[0])

    num_clusters = 0
    rows_clustered = 0
    clusters = -1 * np.ones(acts.shape[0])
    for i, row in enumerate(unique_rows):
        rows_equal_to_this_code = np.where(np.all(binarized == row, axis=1))[0]

        clusters[rows_equal_to_this_code] = num_clusters
        num_clusters += 1
        rows_clustered += rows_equal_to_this_code.shape[0]

    clusters = get_cluster_merging(embedding, clusters)
    num_clusters = len(np.unique(clusters))

    if verbose:
        print("---- Num clusters: {} ---- Percent clustered: {:.3f} ----".format(num_clusters, 1. * rows_clustered / clusters.shape[0]))


    return num_clusters, clusters