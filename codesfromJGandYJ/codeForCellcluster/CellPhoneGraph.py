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
#9.Chung_LTMG_0.9_0.0_0.0_recon.csv
fileR = open('Result_Expimp0.1.txt','w')
dir = '/u1/home/jghhd/Singlecell/TestSingleCluster/allBench/'
exp = '/T2000_UsingOriginalMatrix/T2000_expression.txt'
dir2 = '/u1/home/jghhd/Singlecell/TestSingleCluster/PhenoGraph/Phresults/imputation_0.1/'
exp2 = '_LTMG_0.1_0.0-0.3-0.1_recon.csv'
dir3 = '/u1/home/jghhd/Singlecell/TestSingleCluster/PhenoGraph/Phresults/PreCellType/'
dirresut = '/u1/home/jghhd/Singlecell/TestSingleCluster/PhenoGraph/Phresults/Exp_impu0.1PhenoGraph/'
#9.Chung_LTMG_0.9_0.0_0.0_recon.csv
for eachData in datasetList:
    #print(dir2 + eachData + exp2)

    data = pd.read_csv(dir2 + eachData + exp2, header=None,sep =',')
    #print(data.values.shape)
    topredata = np.exp(data.values)
    communities, graph, Q = phenograph.cluster(topredata)
    print(graph)
    print(Q)


    labels_pred = communities
    true_label = pd.read_csv(dir + eachData + '/' + eachData.split('.')[-1] + '_cell_label.csv',index_col=0)
    df = pd.DataFrame(communities,index=true_label.index)
    df.to_csv(dirresut+ eachData +'_pre_label.csv')
    labels_true = true_label.values.T[0]
    #print(labels_pred)
    #print(labels_true)
    Silhoutte = silhouette_score(topredata,labels_pred)
    ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(labels_true, labels_pred)
    line = 'Data{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(eachData.split('.')[0], ari, ami, nmi, cs, fms, vms, hs, Silhoutte)
    fileR.write(line)
    print('Data{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '.format(eachData.split('.')[0],ari, ami, nmi, cs, fms, vms, hs , Silhoutte), end='\n')
fileR.close()
print('Done')

#sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
#sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_pred, average_mewawthod='arithmetic')
#sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic'
#sklearn.metrics.completeness_score(labels_true, labels_pred)
#sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, sparse=False)
#sklearn.metrics.v_measure_score(labels_true, labels_pred, beta=1.0)
#sklearn.metrics.homogeneity_score(labels_true, labels_pred)