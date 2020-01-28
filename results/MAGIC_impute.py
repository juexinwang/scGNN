# Analysis using MAGIC method
import magic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from benchmark_util import impute_dropout

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
args = parser.parse_args()

# x = np.concatenate([np.random.uniform(-3, -2, (1000, 40)), np.random.uniform(2, 3, (1000, 40))], axis=0)
if args.discreteTag:
    filename = '/home/wangjue/myprojects/scGNN/data/sc/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/home/wangjue/myprojects/scGNN/data/sc/{}/{}.features.csv'.format(args.datasetName,args.datasetName)
x = pd.read_csv(filename,header=None)
x = x.to_numpy()

featuresOriginal = np.copy(x)
features, dropi, dropj, dropix = impute_dropout(featuresOriginal, rate=float(args.ratio))
x = features

magic_operator = magic.MAGIC(knn=10)
X_magic = magic_operator.fit_transform(x, genes="all_genes")
recon = X_magic

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

np.save('/home/wangjue/myprojects/scGNN/otherResults/MAGIC_k10/{}_{}_recon.npy'.format(datasetNameStr,args.ratio),recon)
np.save('/home/wangjue/myprojects/scGNN/otherResults/MAGIC_k10/{}_{}_featuresOriginal.npy'.format(datasetNameStr,args.ratio),featuresOriginal)
np.save('/home/wangjue/myprojects/scGNN/otherResults/MAGIC_k10/{}_{}_dropi.npy'.format(datasetNameStr,args.ratio),dropi)
np.save('/home/wangjue/myprojects/scGNN/otherResults/MAGIC_k10/{}_{}_dropj.npy'.format(datasetNameStr,args.ratio),dropj)
np.save('/home/wangjue/myprojects/scGNN/otherResults/MAGIC_k10/{}_{}_dropix.npy'.format(datasetNameStr,args.ratio),dropix)
