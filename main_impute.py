from __future__ import print_function
import argparse
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
# import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from model import AE, VAE, VAE2d
from util_function import *
from graph_function import *

# Need to change later
parser = argparse.ArgumentParser(description='Only for imputation AutoEncoder-EM for scRNA')
parser.add_argument('--datasetName', type=str, default='MMPbasal_LTMG',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/2.Yan/5.Pollen/MPPbasal/MPPbasal_all/MPPbasal_allgene/MPPbasal_allcell/MPPepo/MMPbasal_LTMG/MMPbasal_all_LTMG')
parser.add_argument('--batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--regulized-type', type=str, default='noregu',
                    help='regulized type (default: Graph), otherwise: noregu')
parser.add_argument('--discreteTag', type=bool, default=False,
                    help='False/True')
parser.add_argument('--model', type=str, default='AE',
                    help='VAE/AE')
parser.add_argument('--npyDir', type=str, default='npyImpute/',
                    help='save npy results in directory')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dropoutRatio', type=float, default=0.1,
                    help='dropout ratio for impute')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)

scData = scDatasetDropout(args.datasetName, args.discreteTag, args.dropoutRatio)
train_loader = DataLoader(scData, batch_size=args.batch_size, shuffle=True, **kwargs)

# Original
if args.model == 'VAE':
    # model = VAE(dim=scData.features.shape[1]).to(device)
    model = VAE2d(dim=scData.features.shape[1]).to(device)
elif args.model == 'AE':
    model = AE(dim=scData.features.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#TODO: batch needs to implement
def train(epoch, train_loader=train_loader, forceReguFlag=False):
    model.train()
    train_loss = 0
    # for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        optimizer.zero_grad()
        if args.model == 'VAE':
            recon_batch, mu, logvar, z = model(data)
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            if forceReguFlag:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, adjsample, adjfeature, 'Graph', args.model)
            else:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, adjsample, adjfeature, args.regulized_type, args.model)
        elif args.model == 'AE':
            recon_batch, z = model(data)
            mu_dummy = ''
            logvar_dummy = ''
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            if forceReguFlag:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, adjsample, adjfeature, 'Graph', args.model)
            else:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, adjsample, adjfeature, args.regulized_type, args.model)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return recon_batch, data, z

# TODO
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    discreteStr = ''
    if args.discreteTag:
        discreteStr = 'D'
    # pca = PCA(n_components=100)
    # pca_result = pca.fit_transform(scData.features.todense())
    # # adj, edgeList = generateAdj(scData.features, graphType='KNNgraph', para = 'cosine:5')
    # # adj, edgeList = generateAdj(pca_result, graphType='KNNgraphPairwise', para = 'Pairwise:10')
    # # adj, edgeList = generateAdj(pca_result, graphType='KNNgraphThreshold', para = 'cosine:10:0.5')
    # adj, edgeList = generateAdj(pca_result, graphType='KNNgraphML', para = 'euclidean:10')
    # adjdense = sp.csr_matrix.todense(adj)
    # adjsample = torch.from_numpy(adjdense)
    # adjsample = adjsample.type(torch.FloatTensor)
    # np.save(args.datasetName+'_'+args.regulized_type+discreteStr+'_edgeList_init.npy',edgeList)       
    adjsample = None
    adjfeature = None

    # save for imputation
    # TODO
    save_sparse_matrix(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
    # sp.save_npz(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
    # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npy',scData.features)
    np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropi.npy',scData.i)
    np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropj.npy',scData.j)
    np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropix.npy',scData.ix)
    

    for epoch in range(1, args.epochs + 1):
        recon, original, z = train(epoch, forceReguFlag=False)
        # test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()
        #     save_image(sample.view(64, 1, 28, 28),
        #                'results/sample_' + str(epoch) + '.png')
    reconOut = recon.detach().cpu().numpy()
    # originalOut = original.detach().cpu().numpy()
    zOut = z.detach().cpu().numpy()   
    np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_recon.npy',reconOut)
    # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_original.npy',originalOut)
    np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_z.npy',zOut)

    adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = 'euclidean:10')
    adjdense = sp.csr_matrix.todense(adj)
    adjsample = torch.from_numpy(adjdense)

    np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_edgeList.npy',edgeList)

    for bigepoch in range(0, 1):
        scDataInter = scDatasetInter(recon)
        train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=True, **kwargs)
        for epoch in range(1, args.epochs + 1):
            recon, original, z = train(epoch, forceReguFlag=True)
        
        reconOut = recon.detach().cpu().numpy()
        # originalOut = original.detach().cpu().numpy()
        zOut = z.detach().cpu().numpy()
        discreteStr = ''
        if args.discreteTag:
            discreteStr = 'D'
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_recon'+str(bigepoch)+'.npy',reconOut)
        # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_original'+str(bigepoch)+'.npy',originalOut)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_z'+str(bigepoch)+'.npy',zOut)

        adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = 'euclidean:10')
        adjdense = sp.csr_matrix.todense(adj)
        adjsample = torch.from_numpy(adjdense)

    np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_edgeList_final.npy',edgeList)

