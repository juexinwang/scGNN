import phenograph
import pandas as pd
import os
import argparse
import numpy as np
from sklearn.metrics import *

def measureClusteringTrueLabel(labels_true, labels_pred):
    '''
    Measure clustering with true labels
    return:
    Adjusted Rand Index, Ajusted Mutual Information, Normalized Mutual Information, completeness score, fowlkes mallows score, v measure score, homogeneity score

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html

    '''
    # sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
    # sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_pred, average_mewawthod='arithmetic')
    # sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'
    # sklearn.metrics.completeness_score(labels_true, labels_pred)
    # sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, sparse=False)
    # sklearn.metrics.v_measure_score(labels_true, labels_pred, beta=1.0)
    # sklearn.metrics.homogeneity_score(labels_true, labels_pred)

    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    cs  = completeness_score(labels_true, labels_pred)
    fms = fowlkes_mallows_score(labels_true, labels_pred)
    vms = v_measure_score(labels_true, labels_pred)
    hs  = homogeneity_score(labels_true, labels_pred)
    return ari, ami, nmi, cs, fms, vms, hs

'''
datasetList = [
    '1.Biase',
    '2.Li',
    '3.Treutlein',
    '4.Yan',
    '5.Goolam',
    '6.Guo',
    '7.Deng',
    '8.Pollen',
    '9.Chung',
    '10.Usoskin',
    '11.Kolodziejczyk',
    '12.Klein',
    '13.Zeisel'
    ]'''
datasetList = [
    '9.Chung',
    '11.Kolodziejczyk',
    '12.Klein',
    '13.Zeisel']
fileR = open('Result_Expimp0.1.txt','w')
dir = '/u1/home/jghhd/Singlecell/TestSingleCluster/allBench/'
exp = '/T2000_UsingOriginalMatrix/T2000_expression.txt'
#predir = '/u1/home/jghhd/Singlecell/TestSingleCluster/Monocle3/Monocle3Pre/'
predir = '/u1/home/jghhd/Singlecell/TestSingleCluster/Monocle3/ExpMonocle3Pre/'
dir2 = '/u1/home/jghhd/Singlecell/TestSingleCluster/PhenoGraph/Phresults/imputation_0.1/'
exp2 = '_LTMG_0.1_0.0-0.3-0.1_recon.csv'
for eachData in datasetList:

    data = pd.read_csv(dir2 + eachData + exp2, header=None,sep =',')
    #communities, graph, Q = phenograph.cluster(data.values.T)
    #labels_pred = communities
    #data = pd.read_csv(predir + eachData.split('.')[-1] + '_cell_pre_PCA.csv', index_col=0)
    #print(data.values.shape)
    topredata = np.exp(data.values)
    labels_pred = pd.read_csv(predir + eachData.split('.')[-1] + '_cell_pre_label.csv',index_col=0)
    true_label = pd.read_csv(dir + eachData + '/' + eachData.split('.')[-1] + '_cell_label.csv',index_col=0)
    labels_true = true_label.values.T[0]
    labels_pred = labels_pred.values.T[0]
    #print(labels_true)
    #print(labels_pred)
    Silhoutte = silhouette_score(topredata, labels_true)
    ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(labels_true, labels_pred)
    line = 'Data{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(eachData.split('.')[0], ari, ami, nmi,
                                                                                 cs, fms, vms, hs, Silhoutte)
    fileR.write(line)
    print(
        'Data{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(eachData.split('.')[0], ari, ami, nmi,
                                                                                 cs, fms, vms, hs, Silhoutte), end='\n')
fileR.close()
#print('Done')



#sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
#sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_pred, average_mewawthod='arithmetic')
#sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'
#sklearn.metrics.completeness_score(labels_true, labels_pred)
#sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, sparse=False)
#sklearn.metrics.v_measure_score(labels_true, labels_pred, beta=1.0)
#sklearn.metrics.homogeneity_score(labels_true, labels_pred)