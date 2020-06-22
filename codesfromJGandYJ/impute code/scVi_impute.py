import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, RetinaDataset, CsvDataset
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer
import torch
import csv
import argparse
import sys
sys.path.append('../')
sys.path.append('/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scGNN-master/')
from benchmark_util import impute_dropout


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, default='data1',help='data1,2,3')
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
parser.add_argument('--outfolder', type=str, default='/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scvi/',
                    help='output filefolder')
args = parser.parse_args()

# Ref:
# https://nbviewer.jupyter.org/github/YosefLab/scVI/blob/master/tests/notebooks/data_loading.ipynb

if args.discreteTag:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scData/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/{}/{}_LTMG_0.1_features.npy'.format(args.data,args.datasetName)
filenameFull = filename
save_path = '/storage/hpc/scratch/yjiang/SCwangjuexin/scGNN-master_021720/scvi/{}/'.format(args.data)

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

x = np.load(filename,allow_pickle=True)
x = x.tolist()
x=x.todense()
x=np.asarray(x)
x=np.log(x+1)


featuresOriginal = np.copy(x)
features, dropi, dropj, dropix = impute_dropout(featuresOriginal, rate=float(args.ratio))

#transpose and add names for rows and cols
features=np.transpose(features)
rowname=np.linspace(1,features.shape[0],features.shape[0]).reshape([features.shape[0],1])
features=np.concatenate([rowname,features],axis=1)
colname=np.linspace(1,features.shape[1],features.shape[1]).reshape([1,features.shape[1]])
features=np.concatenate([colname,features],axis=0)

#write
dropout_filename = save_path+datasetNameStr+"_dropout.csv"
with open(dropout_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerows(features)

# gene_dataset = CortexDataset(save_path=save_path, total_genes=558)
gene_dataset = CsvDataset(dropout_filename, save_path=save_path+args.data+"/")

n_epochs = 400 
lr = 1e-3
use_batches = False
use_cuda = True 

vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
trainer = UnsupervisedTrainer(
    vae,
    gene_dataset,
    train_size=0.75,
    use_cuda=use_cuda,
    frequency=5,
)

trainer.train(n_epochs=n_epochs, lr=lr)


full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()

# use imputation
imputed_values = full.sequential().imputation()
normalized_values = full.sequential().get_sample_scale()

np.save(save_path+'{}_{}_recon.npy'.format(datasetNameStr,args.ratio),imputed_values)
np.save(save_path+'{}_{}_recon_normalized.npy'.format(datasetNameStr,args.ratio),normalized_values)
np.save(save_path+'{}_{}_featuresOriginal.npy'.format(datasetNameStr,args.ratio),featuresOriginal)
np.save(save_path+'{}_{}_dropi.npy'.format(datasetNameStr,args.ratio),dropi)
np.save(save_path+'{}_{}_dropj.npy'.format(datasetNameStr,args.ratio),dropj)
np.save(save_path+'{}_{}_dropix.npy'.format(datasetNameStr,args.ratio),dropix)

# celltype:
#np.save(save_path+'{}_{}_z.npy'.format(datasetNameStr,args.ratio),latent)
