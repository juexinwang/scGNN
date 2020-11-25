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

# pip install scvi==0.6.3
parser = argparse.ArgumentParser(description='scVi imputation')
# In this script, not using arguments
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--ratio', type=str, default='0.1', help='dropoutratio')
args = parser.parse_args()

# Ref:
# https://nbviewer.jupyter.org/github/YosefLab/scVI/blob/master/tests/notebooks/data_loading.ipynb


save_path = '/storage/htc/joshilab/wangjue/scGNN/tmp/'

def impute_scvi(seed=1, datasetName='9.Chung', ratio=0.1):
    filename = '/storage/htc/joshilab/wangjue/scGNN/npyImputeG2E_{}/{}_LTMG_{}_10-0.1-0.9-0.0-0.3-0.1_features.npy'.format(seed, datasetName, ratio)

    x = np.load(filename,allow_pickle=True)
    x = x.tolist()
    x=x.todense()
    x=np.asarray(x)
    x=np.log(x+1)

    features = np.copy(x)

    #transpose and add names for rows and cols
    features=np.transpose(features)
    rowname=np.linspace(1,features.shape[0],features.shape[0]).reshape([features.shape[0],1])
    features=np.concatenate([rowname,features],axis=1)
    colname=np.linspace(1,features.shape[1],features.shape[1]).reshape([1,features.shape[1]])
    features=np.concatenate([colname,features],axis=0)

    #write
    dropout_filename = save_path+"scvi.csv"
    with open(dropout_filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(features)

    # gene_dataset = CortexDataset(save_path=save_path, total_genes=558)
    gene_dataset = CsvDataset(dropout_filename, save_path=save_path)

    n_epochs = 400 
    lr = 1e-3
    use_batches = False
    use_cuda = False 

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

    np.save('/storage/htc/joshilab/wangjue/scGNN/scvi/{}_{}_{}_recon.npy'.format(datasetName,ratio,seed),imputed_values)
    np.save('/storage/htc/joshilab/wangjue/scGNN/scvi/{}_{}_{}_recon_normalized.npy'.format(datasetName,ratio,seed),normalized_values)


datasetNameList = ['9.Chung','11.Kolodziejczyk','12.Klein','13.Zeisel']
seedList = ['1','2','3']
ratioList = [0.1, 0.3, 0.6, 0.8]

for datasetName in datasetNameList:
    for seed in seedList:
        for ratio in ratioList:        
            impute_scvi(seed=seed, datasetName=datasetName, ratio=ratio)

# celltype:
#np.save(save_path+'{}_{}_z.npy'.format(datasetNameStr,args.ratio),latent)
