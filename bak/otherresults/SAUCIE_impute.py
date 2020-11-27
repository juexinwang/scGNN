from model import SAUCIE
from loader import Loader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
sys.path.append('../')
from benchmark_util import impute_dropout

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasetName', type=str, default='MMPbasal',help='MMPbasal_2000')
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

load = Loader(x, shuffle=False)

saucie = SAUCIE(x.shape[1], lambda_c=.2, lambda_d=.4)

saucie.train(load, 500)
embedding = saucie.get_embedding(load)
num_clusters, clusters = saucie.get_clusters(load)
recon = saucie.get_reconstruction(load)

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

# l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax = imputation_error(recon, featuresOriginal, None, dropi, dropj, dropix)
# print('{:.4f} {:.4f} {:.4f} {:.4f} '.format(l1ErrorMean, l1ErrorMedian, l1ErrorMin, l1ErrorMax), end='')

np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE_I/{}_{}_clusters.npy'.format(datasetNameStr,args.ratio),clusters)
np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE_I/{}_{}_recon.npy'.format(datasetNameStr,args.ratio),recon)
np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE_I/{}_{}_featuresOriginal.npy'.format(datasetNameStr,args.ratio),featuresOriginal)
np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE_I/{}_{}_dropi.npy'.format(datasetNameStr,args.ratio),dropi)
np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE_I/{}_{}_dropj.npy'.format(datasetNameStr,args.ratio),dropj)
np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE_I/{}_{}_dropix.npy'.format(datasetNameStr,args.ratio),dropix)

