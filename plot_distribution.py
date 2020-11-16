import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Infer Spatial from Expression in single cells')

parser.add_argument('--datasetName', type=str, default='1.Biase',
                    help='Dataset: 1-13 benchmark: 1.Biase/2.Li/3.Treutlein/4.Yan/5.Goolam/6.Guo/7.Deng/8.Pollen/9.Chung/10.Usoskin/11.Kolodziejczyk/12.Klein/13.Zeisel')
parser.add_argument('--para', type=str, default='LTMG_0.1_10-0.1-0.9-0.0-0.3-0.1',
                    help='save npy results in directory')
parser.add_argument('--inDir', type=str, default='npyGraphTest/',
                    help='save npy results in directory')
parser.add_argument('--outDir', type=str, default='DistNpy/',
                    help='save npy results in directory')
args = parser.parse_args()


ix=np.load(args.inDir+args.datasetName+'_'+args.para+'_dropix.npy')
i =np.load(args.inDir+args.datasetName+'_'+args.para+'_dropi.npy')
j =np.load(args.inDir+args.datasetName+'_'+args.para+'_dropj.npy')
recon   =np.load(args.inDir+args.datasetName+'_'+args.para+'_recon.npy',allow_pickle=True)
features=np.load(args.inDir+args.datasetName+'_'+args.para+'_features.npy',allow_pickle=True)
features=features.tolist()
features=features.todense()

# Directly use plt histogram
# _ = plt.hist(features.ravel())
# plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_features.png')
# plt.close()

# features_log = np.log(features+1)
# _ = plt.hist(features_log.ravel(),bin=100)
# plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_features_log.png')
# plt.close()

# _ = plt.hist(recon.ravel(),bin=100)
# plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_recon.png')
# plt.close()

# recon_exp = np.exp(recon)-1
# plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_recon_exp.png')
# plt.close()

# Use numpy histogram
hist, bin_edges = np.histogram(features.ravel(), bins = np.arange(0,np.max(features),100))
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_features.png')
plt.close()

features_log = np.log(features+1)
hist, bin_edges = np.histogram(features_log.ravel(), bins = np.arange(0,np.max(features),0.1))
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_features_log.png')
plt.close()

hist, bin_edges = np.histogram(recon.ravel(), bins = np.arange(0,np.max(recon),0.1))
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_recon.png')
plt.close()

recon_exp = np.exp(recon)-1
hist, bin_edges = np.histogram(recon_exp.ravel(), bins = np.arange(0,np.max(features),0.1))
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.savefig(args.outDir+'/'+args.datasetName+'_'+args.para+'_recon_exp.png')
plt.close()