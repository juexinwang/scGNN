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
from gae_embedding import GAEembedding,measure_clustering_results,test_clustering_benchmark_results

parser = argparse.ArgumentParser(description='Graph EM AutoEncoder for scRNA')
parser.add_argument('--datasetName', type=str, default='13.Zeisel',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/MMPbasal/MMPbasal_all/MMPbasal_allgene/MMPbasal_allcell/MMPepo/MMPbasal_LTMG/MMPbasal_all_LTMG/MMPbasal_2000')
# Dataset: 1-13 benchmark: 1.Biase/2.Li/3.Treutlein/4.Yan/5.Goolam/6.Guo/7.Deng/8.Pollen/9.Chung/10.Usoskin/11.Kolodziejczyk/12.Klein/13.Zeisel
parser.add_argument('--batch-size', type=int, default=12800, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--EM-iteration', type=int, default=10, metavar='N',
                    help='number of epochs in EM iteration (default: 3)')
parser.add_argument('--EMtype', type=str, default='EM',
                    help='EM process type (default: celltypeEM) or EM')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
parser.add_argument('--converge-type', type=str, default='celltype',
                    help='type of converge: celltype/graph (default: celltype) ')
parser.add_argument('--converge-graphratio', type=float, default=0.001,
                    help='ratio of cell type change in EM iteration (default: 0.001), 0-1')
parser.add_argument('--converge-celltyperatio', type=float, default=0.99,
                    help='ratio of cell type change in EM iteration (default: 0.99), 0-1')
parser.add_argument('--celltype-epochs', type=int, default=200, metavar='N',
                    help='number of epochs in celltype training (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--regulized-type', type=str, default='LTMG',
                    help='regulized type (default: Graph) in EM, otherwise: noregu/LTMG/LTMG01')
parser.add_argument('--discreteTag', action='store_true', default=False, 
                    help='whether input is raw or 0/1 (default: False)')
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type (default: euclidean)')                    
parser.add_argument('--model', type=str, default='AE',
                    help='VAE/AE (default: AE)')
parser.add_argument('--zerofillFlag', action='store_true', default=False, 
                    help='fill zero or not before EM process (default: False)')

#Debug related
parser.add_argument('--saveFlag', action='store_true', default=True, 
                    help='whether save npy results or not')
parser.add_argument('--npyDir', type=str, default='npyGraphTest/',
                    help='save npy results in directory')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
#Clustering related
parser.add_argument('--useGAEembedding', action='store_true', default=False, 
                    help='whether use GAE embedding for clustering(default: False)')
parser.add_argument('--useBothembedding', action='store_true', default=False, 
                    help='whether use both embedding and Graph embedding for clustering(default: False)')
parser.add_argument('--clustering-method', type=str, default='Louvain',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/Birch')
parser.add_argument('--maxClusterNumber', type=int, default=100,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)') 
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')

#GAE related
parser.add_argument('--GAEmodel', type=str, default='gcn_vae', help="models used")
parser.add_argument('--GAEepochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
parser.add_argument('--n-clusters', default=20, type=int, help='number of clusters, 7 for cora, 6 for citeseer, 11 for 5.Pollen, 20 for MMP')
#Start Impute or not, only used for evaluating Impute
parser.add_argument('--imputeMode', default=False, action='store_true',
                    help='impute or not (default: False). Caution: usually change npuDir if set imputeMode as true')
parser.add_argument('--dropoutRatio', type=float, default=0.1,
                    help='dropout ratio for impute (default: 0.1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#TODO
#As we have lots of parameters, should check args
checkargs(args)

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
print(args)

if not args.imputeMode:
    if args.discreteTag:
        scData = scDataset(args.datasetName, args.discreteTag)
    else:
        scData = scDataset(args.datasetName, args.discreteTag, transform=log)
else:
    if args.discreteTag:
        scData = scDatasetDropout(args.datasetName, args.discreteTag, args.dropoutRatio)
    else:
        scData = scDatasetDropout(args.datasetName, args.discreteTag, args.dropoutRatio, transform=log)
train_loader = DataLoader(scData, batch_size=args.batch_size, shuffle=False, **kwargs)

regulationMatrix = readLTMG(args.datasetName)
regulationMatrix = torch.from_numpy(regulationMatrix)

# Original
if args.model == 'VAE':
    # model = VAE(dim=scData.features.shape[1]).to(device)
    model = VAE2d(dim=scData.features.shape[1]).to(device)
elif args.model == 'AE':
    model = AE(dim=scData.features.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#TODO: have to improve save npy
def train(epoch, train_loader=train_loader, EMFlag=False):
    '''
    EMFlag indicates whether in EM processes. 
        If in EM, use regulized-type parsed from program entrance,
        Otherwise, noregu
    '''
    model.train()
    train_loss = 0 
    # for batch_idx, (data, _) in enumerate(train_loader):
    # for batch_idx, data in enumerate(train_loader):
    for batch_idx, (data, dataindex) in enumerate(train_loader):
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        regulationMatrixBatch = regulationMatrix[dataindex,:]
        optimizer.zero_grad()
        if args.model == 'VAE':
            recon_batch, mu, logvar, z = model(data)
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            if EMFlag:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, adjsample, adjfeature, regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, modelusage=args.model)
            else:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, adjsample, adjfeature, regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, modelusage=args.model)
            
        elif args.model == 'AE':
            recon_batch, z = model(data)
            mu_dummy = ''
            logvar_dummy = ''
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            if EMFlag:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, adjsample, adjfeature, regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, modelusage=args.model)
            else:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, adjsample, adjfeature, regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, modelusage=args.model)
               
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        # for batch        
        if batch_idx == 0:
            recon_batch_all=recon_batch 
            data_all = data 
            z_all = z
        else:
            recon_batch_all=torch.cat((recon_batch_all, recon_batch), 0) 
            data_all = torch.cat((data_all, data), 0) 
            z_all = torch.cat((z_all,z),0)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return recon_batch_all, data_all, z_all

if __name__ == "__main__":
    start_time = time.time()
    discreteStr = ''
    if args.discreteTag:
        discreteStr = 'D'       
    # adjsample refer to cell-cell regulization, now we only have adjsample
    adjsample = None
    # adjfeature refer to gene-gene regulization
    adjfeature = None

    # Save results only when impute
    if args.imputeMode:
        # Does not need now
        # save_sparse_matrix(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
        # sp.save_npz(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
        # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npy',scData.features)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropi.npy',scData.i)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropj.npy',scData.j)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_dropix.npy',scData.ix)

    for epoch in range(1, args.epochs + 1):
        recon, original, z = train(epoch, EMFlag=False)
        
    zOut = z.detach().cpu().numpy() 

    prune_time = time.time()        
    # Here para = 'euclidean:10'
    adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
    adjdense = sp.csr_matrix.todense(adj)
    adjsample = torch.from_numpy(adjdense)
    print("---Pruning takes %s seconds ---" % (time.time() - prune_time))
    if args.saveFlag:
        reconOut = recon.detach().cpu().numpy()
        if args.imputeMode:
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_recon.npy',reconOut)
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_z.npy',zOut)
        else:  
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_recon.npy',reconOut)
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_z.npy',zOut)
    
    # Whether use GAE embedding
    if args.useGAEembedding or args.useBothembedding:
        zDiscret = zOut>np.mean(zOut,axis=0)
        zDiscret = 1.0*zDiscret
        if args.useGAEembedding:
            zOut=GAEembedding(zDiscret, adj, args)
        elif args.useBothembedding:
            zEmbedding=GAEembedding(zDiscret, adj, args)
            zOut=np.concatenate((zOut,zEmbedding),axis=1)
        prune_time = time.time()
        # Here para = 'euclidean:10'
        adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
        adjdense = sp.csr_matrix.todense(adj)
        adjsample = torch.from_numpy(adjdense)
        print("---Pruning takes %s seconds ---" % (time.time() - prune_time))
        if args.saveFlag:
            if args.imputeMode:
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_zGAE.npy',zOut)
            else:
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_zGAE.npy',zOut)
        # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_init_edgeList.npy',edgeList)
    
    # For iteration studies
    G0 = nx.Graph()
    G0.add_weighted_edges_from(edgeList)
    nlG0=nx.normalized_laplacian_matrix(G0)
    # set iteration criteria for converge
    adjOld = nlG0
    # set celltype criteria for converge
    listResultOld = [1 for i in range(zOut.shape[0])]

    #Fill the zeros before EM iteration
    # TODO: better implementation later, now we don't filling zeros for now
    if args.zerofillFlag:
        for nz_index in range(len(scData.nz_i)):
            # tmp = scipy.sparse.lil_matrix.todense(scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]])
            # tmp = np.asarray(tmp).reshape(-1)[0]
            tmp = scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]]
            reconOut[scData.nz_i[nz_index], scData.nz_j[nz_index]] = tmp
        recon = reconOut

    print("---Before EM process, proceeded %s seconds ---" % (time.time() - start_time))
    print("EM processes started")
    for bigepoch in range(0, args.EM_iteration):
        iteration_time = time.time()

        # Now for both methods, we need do clustering, using clustering results to check converge
        # TODO May reimplement later
        # Clustering: Get cluster
        clustering_time = time.time()
        if args.clustering_method=='Louvain':
            # Louvain: the only function has R dependent
            # Seperate here for platforms without R support
            from R_util import generateLouvainCluster
            listResult,size = generateLouvainCluster(edgeList)
        elif args.clustering_method=='KMeans':
            clustering = KMeans(n_clusters=args.n_clusters, random_state=0).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method=='SpectralClustering':
            clustering = SpectralClustering(n_clusters=args.n_clusters, assign_labels="discretize", random_state=0).fit(zOut)
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
        print("---Clustering takes %s seconds ---" % (time.time() - clustering_time))

        # If clusters more than maxclusters, then have to stop
        if len(set(listResult))>args.maxClusterNumber or len(set(listResult))<=1:
            print("Stopping: Number of clusters is " + str(len(set(listResult))) + ".")
            # Exit
            # return None
            # Else: dealing with the number
            listResult = trimClustering(listResult,minMemberinCluster=args.minMemberinCluster,maxClusterNumber=args.maxClusterNumber)
        
        #Calculate silhouette
        measure_clustering_results(zOut, listResult)
        print('Total Cluster Number: '+str(len(set(listResult))))

        #Graph regulizated EM AE with celltype AE, do the additional AE
        if args.EMtype == 'celltypeEM': 
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
                train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)
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
        train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)

        for epoch in range(1, args.epochs + 1):
            recon, original, z = train(epoch, EMFlag=True)
        
        zOut = z.detach().cpu().numpy()

        prune_time = time.time()
        # Here para = 'euclidean:10'
        adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
        adjdense = sp.csr_matrix.todense(adj)
        adjsample = torch.from_numpy(adjdense)
        print("---Pruning takes %s seconds ---" % (time.time() - prune_time))

        # Whether use GAE embedding
        if args.useGAEembedding or args.useBothembedding:
            zDiscret = zOut>np.mean(zOut,axis=0)
            zDiscret = 1.0*zDiscret
            if args.useGAEembedding:
                zOut=GAEembedding(zDiscret, adj, args)
            elif args.useBothembedding:
                zEmbedding=GAEembedding(zDiscret, adj, args)
                zOut=np.concatenate((zOut,zEmbedding),axis=1)
            prune_time = time.time()
            # Here para = 'euclidean:10'
            adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
            adjdense = sp.csr_matrix.todense(adj)
            adjsample = torch.from_numpy(adjdense)
            print("---Pruning takes %s seconds ---" % (time.time() - prune_time))

        if args.saveFlag:
            reconOut = recon.detach().cpu().numpy()
            if args.imputeMode:
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_recon'+str(bigepoch)+'.npy',reconOut)
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_z'+str(bigepoch)+'.npy',zOut)
            else:
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_recon'+str(bigepoch)+'.npy',reconOut)
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_z'+str(bigepoch)+'.npy',zOut)
        
        print("---One iteration in EM process, proceeded %s seconds ---" % (time.time() - iteration_time))

        #Iteration usage
        Gc = nx.Graph()
        Gc.add_weighted_edges_from(edgeList)
        adjGc = nx.adjacency_matrix(Gc)
        
        # Update new adj
        adjNew = args.alpha*nlG0 + (1-args.alpha) * adjGc/np.sum(adjGc,axis=0)
        
        #debug
        print('adjNew:{} adjOld:{} threshold:{}'.format(adjNew, adjOld, args.converge_graphratio*nlG0))
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(listResultOld, listResult)
        print(listResultOld)
        print(listResult)
        print('celltype similarity:'+str(ari))
        # graph criteria here
        if args.converge_type == 'graph':       
            if abs(np.mean(adjNew-adjOld)) < args.converge_graphratio * nlG0:
                print('Converge now!')
                break
        # celltype criteria here
        elif args.converge_type == 'celltype':            
            if ari>args.converge_celltyperatio:
                print('Converge now!')
                break 

        # Update
        adjOld = adjNew
        listResultOld = listResult
            
    if args.saveFlag:
        if args.imputeMode:
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_final_edgeList.npy',edgeList)
        else:
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_final_edgeList.npy',edgeList)
    print("---Total Running Time: %s seconds ---" % (time.time() - start_time))
