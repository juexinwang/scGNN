import numpy as np
import argparse
import community
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Plot the lineage by markers in each cell types
# Louvain clustering
# https://python-louvain.readthedocs.io/en/latest/api.html
# https://github.com/taynaud/python-louvain
parser = argparse.ArgumentParser(description='AutoEncoder-EM for scRNA')
parser.add_argument('--datasetName', type=str, default='MPPbasal',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/2.Yan/5.Pollen/MPPbasal/MPPepo')

args = parser.parse_args()

# expressionData = np.load('MPPbasal_noregu_original.npy')
# expressionData = np.reshape(expressionData, (expressionData.shape[0],-1))

expressionData = pd.read_csv('../data/sc/MPPbasal/MPPbasal.features.csv',header=None)
expressionData = expressionData.to_numpy()

markerGeneList = ['Kit','Flt3','Dntt','Ebf1','Cd19','Lmo4','Ms4a2','Ear10','Cd74','Irf8','Mpo','Elane','Ngp','Mpl','Pf4','Car1','Gata1','Hbb-bs','Ptgfrn','Mki67']
markerGeneIndexList = []
geneDict = {}
geneList = []

# with open("data/sc/{}/{}.gene.txt".format(args.datasetName, args.datasetName), 'r') as f:
with open("../data/sc/MPPbasal/MPPbasal.gene.txt", 'r') as f:
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
allIndex = np.where(useData>np.mean(useData,0))

resultTable = [[0.0] * len(markerGeneList)  for i in range(len(set(df_subset['Cluster'])))]

clusterNum = [0 for i in range(len(set(df_subset['Cluster'])))]
for i in range(useData.shape[0]):
    clusterNum[df_subset['Cluster'][i]] += 1

clusterNum = np.asarray(clusterNum).reshape(len(set(df_subset['Cluster'])),1)

for i in np.arange(allIndex[0].shape[0]):
    # print(i)
    # print(allIndex[0][i])
    # print(df_subset['Cluster'][allIndex[0][i]])
    # print("*")
    clusterIndex = df_subset['Cluster'][allIndex[0][i]]
    resultTable[clusterIndex][allIndex[1][i]] += 1

resultTableUsage = resultTable/clusterNum





