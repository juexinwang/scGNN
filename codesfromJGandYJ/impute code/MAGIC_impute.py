# Analysis using MAGIC method
import magic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
#from benchmark_util import impute_dropout

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
args = parser.parse_args()


def impute_Magic(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/hpc/scratch/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)
    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)
    x=np.log(x+1)

    # Load single-cell RNA-seq data
    # Default is KNN=5
    magic_operator = magic.MAGIC()
    # magic_operator = magic.MAGIC(knn=10)
    X_magic = magic_operator.fit_transform(x, genes="all_genes")
    recon = X_magic

    np.save('/storage/hpc/scratch/wangjue/scGNN/magic/{}_{}_{}_recon.npy'.format(datasetName,ratio,seed),recon)

datasetNameList = ['9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel']
seedList = ['1','2','3']
ratioList = [0.1, 0.3, 0.6, 0.8]

for datasetName in datasetNameList:
    for seed in seedList:
        for ratio in ratioList:        
            impute_Magic(seed=seed, datasetName=datasetName, ratio=ratio)

# From scVI
# # Load single-cell RNA-seq data
# scdata = magic.mg.SCData(x, "sc-seq")
# print(scdata)

# scdata.run_magic(n_pca_components=20, random_pca=True, t=6, k=30, ka=10, epsilon=1, rescale_percent=99)

# if len(sys.argv) == 2:
#     np.save("t_MAGIC.npy", scdata.magic.data.as_matrix())
