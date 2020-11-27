# Analysis using SAUCIE method 
from model import SAUCIE
from loader import Loader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasetName', type=str, default='MMPbasal',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False, 
                    help='whether input is raw or 0/1 (default: False)')
args = parser.parse_args()

# x = np.concatenate([np.random.uniform(-3, -2, (1000, 40)), np.random.uniform(2, 3, (1000, 40))], axis=0)
if args.discreteTag:
    filename = '/home/wangjue/myprojects/scGNN/data/sc/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/home/wangjue/myprojects/scGNN/data/sc/{}/{}.features.csv'.format(args.datasetName,args.datasetName)
x = pd.read_csv(filename,header=None)
x = x.to_numpy()

load = Loader(x, shuffle=False)

saucie = SAUCIE(x.shape[1], lambda_c=.2, lambda_d=.4)


saucie.train(load, 500)
embedding = saucie.get_embedding(load)
num_clusters, clusters = saucie.get_clusters(load)

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE/{}_z.npy'.format(datasetNameStr),embedding)
np.save('/home/wangjue/myprojects/scGNN/otherResults/SAUCIE/{}_clusters.npy'.format(datasetNameStr),clusters)


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(embedding[:, 0], embedding[:, 1], c=clusters)
# fig.savefig('embedding_by_cluster.png')
