from __future__ import print_function
import time
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from inspect import signature
import scipy
import pandas as pd
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import umap

import community
import networkx as nx

#Original
df = pd.read_csv('/home/wangjue/biodata/scData/AnjunBenchmark/5.Pollen/Pollen_cell_label.csv')
df.columns = ['Cell','Cluster']
# z = np.load('5.Pollen_noreguD_z.npy')

# z = pd.read_csv('data/sc/5.Pollen/5.Pollen.features.D.csv',header=None)
z = pd.read_csv('data/sc/MPPbasal/MPPbasal.features.D.csv',header=None)
# plt.scatter(z[:,0],z[:,1],c=df['Cluster'],cmap=cm.brg)
# plt.show()

edgeList = np.load('MPPbasal_noreguD_edgeList.npy')
edgeList = edgeList.tolist()
G = nx.Graph(edgeList)
partition = community.best_partition(G)
valueResults = []
for key in partition.keys():
    valueResults.append(partition[key])

df = pd.DataFrame()
df['Cluster']=valueResults


#PCA
pca = PCA(n_components=500)
pca_result = pca.fit_transform(z)
re = pd.DataFrame()
re['pca-one'] = pca_result[:,0]
re['pca-two'] = pca_result[:,1] 
re['pca-three'] = pca_result[:,2]
re['Cluster'] = df['Cluster']
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="Cluster",
    palette=sns.color_palette("brg", 11),
    data=re,
    legend="full",
)

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=re["pca-one"], 
    ys=re["pca-two"], 
    zs=re["pca-three"], 
    c=df['Cluster'], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(z)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['Cluster'] = df['Cluster']
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", 
    y="tsne-2d-two",
    hue="Cluster",
    palette=sns.color_palette("brg",3),
    data=df_subset,
    legend="full",
    # alpha=0.3
)

#UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(z)
plt.scatter(embedding[:, 0], embedding[:, 1], c=df['Cluster'], cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24)


# Louvain clustering
# https://python-louvain.readthedocs.io/en/latest/api.html
# https://github.com/taynaud/python-louvain
import community
import networkx as nx
import matplotlib.pyplot as plt

# Replace this with your networkx graph loading depending on your format !
G = nx.erdos_renyi_graph(30, 0.05)

#first compute the best partition
partition = community.best_partition(G)

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))

nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()



