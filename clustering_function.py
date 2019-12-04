import numpy as np
import argparse
import community
import networkx as nx
import matplotlib.pyplot as plt
# Louvain clustering
# https://python-louvain.readthedocs.io/en/latest/api.html
# https://github.com/taynaud/python-louvain

parser = argparse.ArgumentParser(description='AutoEncoder-EM for scRNA')
parser.add_argument('--datasetName', type=str, default='MPPbasal',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/2.Yan/5.Pollen/MPPbasal/MPPepo')

args = parser.parse_args()



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


nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G,pos)
plt.show()


expressionData = np.load('test/MPPbasal_noregu_recon.npy')

markerGeneList = ['Kit','Flt3','Dntt','Ebf1','Cd19','Lmo4','Ms4a2','Ear10','Cd74','Irf8','Mpo','Elane','Ngp','Mpl','Pf4','Car1','Gata1','Hbb-bs','Ptgfrn','Mki67']
markerGeneIndexList = []
geneDict = {}
geneList = []

# with open("data/sc/{}/{}.gene.txt".format(args.datasetName, args.datasetName), 'r') as f:
with open("data/sc/MPPbasal/MPPbasal.gene.txt", 'r') as f:
    lines = f.readlines()
    count = 0
    for line in lines:
        line = line.strip()
        geneList.append(line)
        geneDict[line]=count
        count += 1
f.close()

for markerGene in markerGeneList:
    if markerGene in geneDict:
        markerGeneIndexList.append(geneDict[markerGene])
    else:
        print('Cannot find '+markerGene+ ' in gene.txt')

useData = expressionData[:,markerGeneIndexList]



