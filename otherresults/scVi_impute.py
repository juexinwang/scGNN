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
from benchmark_util import impute_dropout

parser = argparse.ArgumentParser(description='')
parser.add_argument('--datasetName', type=str, default='MMPbasal_2000',help='MMPbasal_2000')
parser.add_argument('--discreteTag', action='store_true', default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--ratio', type=str, default='0.1',
                    help='dropoutratio')
parser.add_argument('--filefolder', type=str, default='/home/wangjue/myprojects/scGNN/otherResults/scVi/',
                    help='output filefolder')
args = parser.parse_args()

# Ref:
# https://nbviewer.jupyter.org/github/YosefLab/scVI/blob/master/tests/notebooks/data_loading.ipynb

if args.discreteTag:
    filename = '/home/wangjue/myprojects/scGNN/data/sc/{}/{}.features.D.csv'.format(args.datasetName,args.datasetName)
else:
    filename = '/home/wangjue/myprojects/scGNN/data/sc/{}/{}.features.csv'.format(args.datasetName,args.datasetName)
save_path = args.filefolder

x = pd.read_csv(filename,header=None)
x = x.to_numpy()

featuresOriginal = np.copy(x)
features, dropi, dropj, dropix = impute_dropout(featuresOriginal, rate=float(args.ratio))

#write
dropout_filename = save_path+"output.csv"
with open(dropout_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerows(features)

# gene_dataset = CortexDataset(save_path=save_path, total_genes=558)
gene_dataset = CsvDataset(dropout_filename, save_path=save_path)

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
# if os.path.isfile('%s/vae.pkl' % save_path):
#     trainer.model.load_state_dict(torch.load('%s/vae.pkl' % save_path))
#     trainer.model.eval()
# else:
#     trainer.train(n_epochs=n_epochs, lr=lr)
#     torch.save(trainer.model.state_dict(), '%s/vae.pkl' % save_path)

#plot
# elbo_train_set = trainer.history["elbo_train_set"]
# elbo_test_set = trainer.history["elbo_test_set"]
# x = np.linspace(0, 500, (len(elbo_train_set)))
# plt.plot(x, elbo_train_set)
# plt.plot(x, elbo_test_set)
# plt.ylim(1150, 1600)

full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
latent, batch_indices, labels = full.sequential().get_latent()
batch_indices = batch_indices.ravel()

# use imputation
imputed_values = full.sequential().imputation()
# normalized_values = full.sequential().get_sample_scale()

discreteStr = ''
if args.discreteTag:
    discreteStr = 'D'
datasetNameStr = args.datasetName+discreteStr

np.save(save_path+'{}_{}_recon.npy'.format(datasetNameStr,args.ratio),imputed_values)
np.save(save_path+'{}_{}_featuresOriginal.npy'.format(datasetNameStr,args.ratio),featuresOriginal)
np.save(save_path+'{}_{}_dropi.npy'.format(datasetNameStr,args.ratio),dropi)
np.save(save_path+'{}_{}_dropj.npy'.format(datasetNameStr,args.ratio),dropj)
np.save(save_path+'{}_{}_dropix.npy'.format(datasetNameStr,args.ratio),dropix)

# celltype:
np.save(save_path+'{}_{}_z.npy'.format(datasetNameStr,args.ratio),latent)