# Analysis using MAGIC method
import magic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
sys.path.append('../')
sys.path.append('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/')
#from benchmark_util import impute_dropout

def impute_dropout(X, rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    """
    #If the input is a dense matrix
    if isinstance(X, np.ndarray):
        X_zero = np.copy(X)
        # select non-zero subset
        i,j = np.nonzero(X_zero)
    # If the input is a sparse matrix
    else:
        X_zero = scipy.sparse.lil_matrix.copy(X)
        # select non-zero subset
        i,j = X_zero.nonzero()
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(np.floor(0.1 * len(i))), replace=False)
    X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
    # choice number 2, focus on a few but corrupt binomially
    #ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    #X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default='data1',help='data1,2,3')
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
args = parser.parse_args()


# x = np.concatenate([np.random.uniform(-3, -2, (1000, 40)), np.random.uniform(2, 3, (1000, 40))], axis=0)
if args.discreteTag:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scData/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/{}/{}_LTMG_0.1_features.npy'.format(args.data,args.datasetName)
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

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

np.save('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/magic/{}/{}_{}_recon.npy'.format(args.data,datasetNameStr,args.ratio),recon)


# From scVI
# # Load single-cell RNA-seq data
# scdata = magic.mg.SCData(x, "sc-seq")
# print(scdata)

# scdata.run_magic(n_pca_components=20, random_pca=True, t=6, k=30, ka=10, epsilon=1, rescale_percent=99)

# if len(sys.argv) == 2:
#     np.save("t_MAGIC.npy", scdata.magic.data.as_matrix())
