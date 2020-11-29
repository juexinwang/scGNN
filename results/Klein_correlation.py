import numpy as np
from scipy import stats
import csv

# Get correlation from gene interactions from Klein datasets
# Ref: Klein, Allon M., et al. "Droplet barcoding for single-cell transcriptomics applied to embryonic stem cells." Cell 161.5 (2015): 1187-1201.

geneList=[
    'Krt8', #4
    'S100a6', #19
    'Id2', #895
    'Id1', #602
    'ld3', #1559
    'Ccnd1',# not in the range
    'Ccnb1',# not in the range
    'Ccnd2',# not in the range
    'Ccna1',# not in the range
    'Sox17',# not in the range
    'Col4a1', #226
    'Pou5f1', #150
    'Ccnd3', #255
    'Ccna2',# not in the range
    'Nanog', #1449
    'Klf4',# not in the range
    'Sox2', # 601
    'Zfp42', #527
    'Trim28', #136
    'Esrrb', #849
    'Tdh', #206
]

geneNumList=[
    4,
    19,
    895,
    602,
    1559,
    226,
    150,
    255,
    1449,
    601,
    527,
    136,
    849,
    206,
]

savedir = './fig3/'
# methodList = ['magic','saucie','saver','scimpute','scvi','scvinorm','dca','deepimpute','scIGANs','netNMFsclog','netNMFsc']
methodList = ['magic','saucie','saver','scimpute','scvi','scvinorm','dca','deepimpute']

def corCal(method='magic'):
    if method == 'scvinorm':
        filename = '/storage/htc/joshilab/wangjue/scGNN/scvi/12.Klein_0.0_1_recon_normalized.npy'
    else:
        filename = '/storage/htc/joshilab/wangjue/scGNN/{}/12.Klein_0.0_1_recon.npy'.format(method)
    x = np.load(filename,allow_pickle=True)
    x = x.T

    corr = np.zeros(len(geneNumList),len(geneNumList))
    for i in range(len(geneNumList)):
        for j in range(len(geneNumList)):
            corr[i,j]=stats.pearsonr(x[geneNumList[i],:], x[geneNumList[j],:])

    out_filename = savedir+method+".csv"
    with open(out_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(corr)


for method in methodList:        
    corCal(method=method)