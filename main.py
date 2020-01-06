from __future__ import print_function
import time
import argparse
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans,SpectralClustering,AffinityPropagation,AgglomerativeClustering,Birch,DBSCAN,FeatureAgglomeration,MeanShift,OPTICS 
from model import AE, VAE, VAE2d
from util_function import *
from graph_function import *
from benchmark_util import *
from gae_embedding import GAEembedding

parser = argparse.ArgumentParser(description='Graph EM AutoEncoder for scRNA')
parser.add_argument('--datasetName', type=str, default='MMPbasal',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/2.Yan/5.Pollen/MPPbasal/MPPbasal_all/MPPbasal_allgene/MPPbasal_allcell/MPPepo/MMPbasal_LTMG/MMPbasal_all_LTMG')
parser.add_argument('--batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--EM-iteration', type=int, default=2, metavar='N',
                    help='number of epochs in EM iteration (default: 3)')
parser.add_argument('--celltype-epochs', type=int, default=2, metavar='N',
                    help='number of epochs in celltype training (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--regulized-type', type=str, default='Graph',
                    help='regulized type (default: Graph) in EM, otherwise: noregu')
parser.add_argument('--discreteTag', type=bool, default=False,
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type (default: euclidean)')                    
parser.add_argument('--model', type=str, default='AE',
                    help='VAE/AE (default: AE)')
parser.add_argument('--zerofillFlag', type=bool, default=False,
                    help='fill zero or not before EM process (default: False)')
parser.add_argument('--EMtype', type=str, default='celltypeEM',
                    help='EM process type (default: celltypeEM) or EM')
#Debug related
parser.add_argument('--saveTag', type=bool, default=False,
                    help='whether save npy results or not')
parser.add_argument('--npyDir', type=str, default='npyGraphTest/',
                    help='save npy results in directory')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
#Clustering related
parser.add_argument('--useGAEembedding', type=bool, default=True,
                    help='whether use GAE embedding before clustering(default: True)')
parser.add_argument('--clustering-method', type=str, default='Louvain',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/Birch')
#GAE related
parser.add_argument('--GAEmodel', type=str, default='gcn_vae', help="models used")
parser.add_argument('--GAEepochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
parser.add_argument('--GAEn-clusters', default=20, type=int, help='number of clusters, 7 for cora, 6 for citeseer, 11 for 5.Pollen, 20 for MMP')
#Start Impute or not, only used for evaluating Impute
parser.add_argument('--imputeTag', type=bool, default=False,
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeTag as true')
parser.add_argument('--dropoutRatio', type=float, default=0.1,
                    help='dropout ratio for impute (default: 0.1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if not args.imputeTag:
    scData = scDataset(args.datasetName, args.discreteTag)
else:
    scData = scDatasetDropout(args.datasetName, args.discreteTag, args.dropoutRatio)
train_loader = DataLoader(scData, batch_size=args.batch_size, shuffle=True, **kwargs)

# Original
if args.model == 'VAE':
    # model = VAE(dim=scData.features.shape[1]).to(device)
    model = VAE2d(dim=scData.features.shape[1]).to(device)
elif args.model == 'AE':
    model = AE(dim=scData.features.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#TODO: needs to implement batch
def train(epoch, train_loader=train_loader, EMFlag=False):
    '''
    EMFlag indicates whether in EM processes. 
        If in EM, use regulized-type parsed from program entrance,
        Otherwise, noregu
    '''
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
            if EMFlag:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, adjsample, adjfeature, args.regulized_type, args.model)
            else:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, adjsample, adjfeature, 'noregu', args.model)
            
        elif args.model == 'AE':
            recon_batch, z = model(data)
            mu_dummy = ''
            logvar_dummy = ''
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            if EMFlag:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, adjsample, adjfeature, args.regulized_type, args.model)
            else:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, adjsample, adjfeature, 'noregu', args.model)
               
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

if __name__ == "__main__":
    discreteStr = ''
    if args.discreteTag:
        discreteStr = 'D'       
    # adjsample refer to cell-cell regulization, now we only have adjsample
    adjsample = None
    # adjfeature refer to gene-gene regulization
    adjfeature = None

    # Save results only when impute
    if args.imputeTag:
        save_sparse_matrix(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
        # sp.save_npz(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
        # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npy',scData.features)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropi.npy',scData.i)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropj.npy',scData.j)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropix.npy',scData.ix)

    for epoch in range(1, args.epochs + 1):
        recon, original, z = train(epoch, EMFlag=False)
        
    zOut = z.detach().cpu().numpy() 
        
    # Here para = 'euclidean:10'
    adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
    adjdense = sp.csr_matrix.todense(adj)
    adjsample = torch.from_numpy(adjdense)
    if args.saveTag:
        reconOut = recon.detach().cpu().numpy()  
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_recon.npy',reconOut)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_z.npy',zOut)
    
    # Whether use GAE embedding
    if args.useGAEembedding:
        zDiscret = zOut>np.mean(zOut,axis=0)
        zDiscret = 1.0*zDiscret
        zOut=GAEembedding(zDiscret, adj)
        # Here para = 'euclidean:10'
        adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
        adjdense = sp.csr_matrix.todense(adj)
        adjsample = torch.from_numpy(adjdense)
        if args.saveTag:
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_zGAE.npy',zOut)
        # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_init_edgeList.npy',edgeList)
    
    #Fill the zeros before EM iteration
    # TODO: better implementation
    if args.zerofillFlag:
        # start_time = time.time()
        for nz_index in range(len(scData.nz_i)):
            # tmp = scipy.sparse.lil_matrix.todense(scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]])
            # tmp = np.asarray(tmp).reshape(-1)[0]
            tmp = scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]]
            reconOut[scData.nz_i[nz_index], scData.nz_j[nz_index]] = tmp
        recon = reconOut
        # print("--- %s seconds ---" % (time.time() - start_time))

    print("EM processes started")
    for bigepoch in range(0, args.EM_iteration):
        #Graph regulizated EM AE with celltype AE, do the additional AE
        if args.EMtype == 'celltypeEM':            
            # Clustering: Get cluster
            if args.clustering_method=='Louvain':
                listResult,size = generateCluster(edgeList)
            elif args.clustering_method=='KMeans':
                clustering = KMeans(n_clusters=args.GAEn_clusters, random_state=0).fit(zOut)
                listResult = clustering.predict(zOut)
            elif args.clustering_method=='SpectralClustering':
                clustering = SpectralClustering(n_clusters=args.GAEn_clusters, assign_labels="discretize", random_state=0).fit(zOut)
                listResult = clustering.labels_.tolist()
            elif args.clustering_method=='AffinityPropagation':
                clustering = AffinityPropagation().fit(zOut)
                listResult = clustering.predict(zOut)
            elif args.clustering_method=='AgglomerativeClustering':
                clustering = AgglomerativeClustering().fit(zOut)
                listResult = clustering.labels_.tolist()
            elif args.clustering_method=='Birch':
                clustering = Birch(n_clusters=args.n_clusters).fit(zOut)
                listResult = clustering.predict(zOut)
            else:
                print("Error: Clustering method not appropriate")

            # Each cluster has a autoencoder, and organize them back in iteraization
            clusterIndexList = []
            for i in range(len(set(listResult))):
                clusterIndexList.append([])
            for i in range(len(listResult)):
                clusterIndexList[listResult[i]].append(i)

            reconNew = np.zeros((scData.features.shape[0],scData.features.shape[1]))
            
            # Convert to Tensor
            reconNew = torch.from_numpy(reconNew)
            reconNew = reconNew.type(torch.FloatTensor)
            reconNew = reconNew.to(device)

            for clusterIndex in clusterIndexList:
                reconUsage = recon[clusterIndex]
                scDataInter = scDatasetInter(reconUsage)
                train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=True, **kwargs)
                for epoch in range(1, args.celltype_epochs + 1):
                    reconCluster, originalCluster, zCluster = train(epoch, EMFlag=True)                
                count = 0
                for i in clusterIndex:
                    reconNew[i] = reconCluster[count,:]
                    count +=1
            # Update
            recon = reconNew
        
        # Use new dataloader
        scDataInter = scDatasetInter(recon)
        train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=True, **kwargs)

        for epoch in range(1, args.epochs + 1):
            recon, original, z = train(epoch, EMFlag=True)
        
        zOut = z.detach().cpu().numpy()

        # Here para = 'euclidean:10'
        adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
        adjdense = sp.csr_matrix.todense(adj)
        adjsample = torch.from_numpy(adjdense)

        # Whether use GAE embedding
        if args.useGAEembedding:
            zDiscret = zOut>np.mean(zOut,axis=0)
            zDiscret = 1.0*zDiscret
            zOut=GAEembedding(zDiscret, adj)
            # Here para = 'euclidean:10'
            adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
            adjdense = sp.csr_matrix.todense(adj)
            adjsample = torch.from_numpy(adjdense)

        if args.saveTag:
            reconOut = recon.detach().cpu().numpy()
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_recon'+str(bigepoch)+'.npy',reconOut)
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_z'+str(bigepoch)+'.npy',zOut)
    
    if args.saveTag:
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_final_edgeList.npy',edgeList)
