import time
import os
import argparse
import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import resource
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN, FeatureAgglomeration, OPTICS, MeanShift
from model import AE, VAE
from util_function import *
from graph_function import *
from benchmark_util import *
from gae_embedding import GAEembedding, measure_clustering_results, test_clustering_benchmark_results
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description='Main entrance of scGNN')
parser.add_argument('--datasetName', type=str, default='481193cb-c021-4e04-b477-0b7cfef4614b.mtx',
                    help='For 10X: folder name of 10X dataset; For CSV: csv file name')
parser.add_argument('--datasetDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='Directory of dataset: default(/home/wangjue/biodata/scData/10x/6/)')

parser.add_argument('--batch-size', type=int, default=12800, metavar='N',
                    help='input batch size for training (default: 12800)')
parser.add_argument('--Regu-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train in Feature Autoencoder initially (default: 500)')
parser.add_argument('--EM-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train Feature Autoencoder in iteration EM (default: 200)')
parser.add_argument('--EM-iteration', type=int, default=10, metavar='N',
                    help='number of iteration in total EM iteration (default: 10)')
parser.add_argument('--quickmode', action='store_true', default=False,
                    help='whether use quickmode, skip Cluster Autoencoder (default: no quickmode)')
parser.add_argument('--cluster-epochs', type=int, default=200, metavar='N',
                    help='number of epochs in Cluster Autoencoder training (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable GPU training. If you only have CPU, add --no-cuda in the command line')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--regulized-type', type=str, default='noregu',
                    help='regulized type (default: LTMG) in EM, otherwise: noregu/LTMG/LTMG01')
parser.add_argument('--reduction', type=str, default='sum',
                    help='reduction type: mean/sum, default(sum)')
parser.add_argument('--model', type=str, default='AE',
                    help='VAE/AE (default: AE)')
parser.add_argument('--gammaPara', type=float, default=0.1,
                    help='regulized intensity (default: 0.1)')
parser.add_argument('--alphaRegularizePara', type=float, default=0.9,
                    help='regulized parameter (default: 0.9)')

# Build cell graph
parser.add_argument('--k', type=int, default=10,
                    help='parameter k in KNN graph (default: 10)')
parser.add_argument('--knn-distance', type=str, default='euclidean',
                    help='KNN graph distance type: euclidean/cosine/correlation (default: euclidean)')
parser.add_argument('--prunetype', type=str, default='KNNgraphStatsSingleThread',
                    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStatsSingleThread)')

# Debug related
parser.add_argument('--precisionModel', type=str, default='Float',
                    help='Single Precision/Double precision: Float/Double (default:Float)')
parser.add_argument('--coresUsage', type=str, default='1',
                    help='how many cores used: all/1/... (default:1)')
parser.add_argument('--outputDir', type=str, default='outputDir/',
                    help='save npy results in directory')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--saveinternal', action='store_true', default=False,
                    help='whether save internal interation results or not')
parser.add_argument('--debugMode', type=str, default='noDebug',
                    help='savePrune/loadPrune for extremely huge data in debug (default: noDebug)')
parser.add_argument('--nonsparseMode', action='store_true', default=False,
                    help='SparseMode for running for huge dataset')

# LTMG related
parser.add_argument('--LTMGDir', type=str, default='/storage/htc/joshilab/wangjue/casestudy/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--ltmgExpressionFile', type=str, default='Use_expression.csv',
                    help='expression File after ltmg in csv')
parser.add_argument('--ltmgFile', type=str, default='LTMG_sparse.mtx',
                    help='expression File in csv. (default:LTMG_sparse.mtx for sparse mode/ ltmg.csv for nonsparse mode) ')

# Clustering related
parser.add_argument('--useGAEembedding', action='store_true', default=False,
                    help='whether use GAE embedding for clustering(default: False)')
parser.add_argument('--useBothembedding', action='store_true', default=False,
                    help='whether use both embedding and Graph embedding for clustering(default: False)')
parser.add_argument('--n-clusters', default=20, type=int,
                    help='number of clusters if predifined for KMeans/Birch ')
parser.add_argument('--clustering-method', type=str, default='LouvainK',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/AgglomerativeClusteringK/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB')
parser.add_argument('--maxClusterNumber', type=int, default=30,
                    help='max cluster for celltypeEM without setting number of clusters (default: 30)')
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')
parser.add_argument('--resolution', type=str, default='auto',
                    help='the number of resolution on Louvain (default: auto/0.5/0.8)')

# imputation related
parser.add_argument('--EMregulized-type', type=str, default='Celltype',
                    help='regulized type (default: noregu) in EM, otherwise: noregu/Graph/GraphR/Celltype')
parser.add_argument('--gammaImputePara', type=float, default=0.0,
                    help='regulized parameter (default: 0.0)')
parser.add_argument('--graphImputePara', type=float, default=0.3,
                    help='graph parameter (default: 0.3)')
parser.add_argument('--celltypeImputePara', type=float, default=0.1,
                    help='celltype parameter (default: 0.1)')
parser.add_argument('--L1Para', type=float, default=1.0,
                    help='L1 regulized parameter (default: 0.001)')
parser.add_argument('--L2Para', type=float, default=0.0,
                    help='L2 regulized parameter (default: 0.001)')
parser.add_argument('--EMreguTag', action='store_true', default=False,
                    help='whether regu in EM process')
parser.add_argument('--sparseImputation', type=str, default='nonsparse',
                    help='whether use sparse in imputation: sparse/nonsparse (default: nonsparse)')

# dealing with zeros in imputation results
parser.add_argument('--zerofillFlag', action='store_true', default=False,
                    help='fill zero or not before EM process (default: False)')
parser.add_argument('--noPostprocessingTag', action='store_false', default=True,
                    help='whether postprocess imputated results, default: (True)')
parser.add_argument('--postThreshold', type=float, default=0.01,
                    help='Threshold to force expression as 0, default:(0.01)')

# Converge related
parser.add_argument('--alpha', type=float, default=0.5,
                    help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
parser.add_argument('--converge-type', type=str, default='celltype',
                    help='type of converge condition: celltype/graph/both/either (default: celltype) ')
parser.add_argument('--converge-graphratio', type=float, default=0.01,
                    help='converge condition: ratio of graph ratio change in EM iteration (default: 0.01), 0-1')
parser.add_argument('--converge-celltyperatio', type=float, default=0.99,
                    help='converge condition: ratio of cell type change in EM iteration (default: 0.99), 0-1')

# GAE related
parser.add_argument('--GAEmodel', type=str,
                    default='gcn_vae', help="models used")
parser.add_argument('--GAEepochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--GAEhidden1', type=int, default=32,
                    help='Number of units in hidden layer 1.')
parser.add_argument('--GAEhidden2', type=int, default=16,
                    help='Number of units in hidden layer 2.')
parser.add_argument('--GAElr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--GAEdropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--GAElr_dw', type=float, default=0.001,
                    help='Initial learning rate for regularization.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.sparseMode = not args.nonsparseMode

# TODO
# As we have lots of parameters, should check args
checkargs(args)

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
print('Using device:'+str(device))

if not args.coresUsage == 'all':
    torch.set_num_threads(int(args.coresUsage))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# print(args)
start_time = time.time()

# load scRNA in csv
print('---0:00:00---scRNA starts loading.')
data, genelist, celllist = loadscExpression(
    args.datasetDir+args.datasetName+'/'+args.ltmgExpressionFile, sparseMode=args.sparseMode)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---scRNA has been successfully loaded')

scData = scDataset(data)
train_loader = DataLoader(
    scData, batch_size=args.batch_size, shuffle=False, **kwargs)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---TrainLoader has been successfully prepared.')

# load LTMG in sparse version
if not args.regulized_type == 'noregu':
    print('Start loading LTMG in sparse coding.')
    regulationMatrix = readLTMG(
        args.LTMGDir+args.datasetName+'/', args.ltmgFile)
    regulationMatrix = torch.from_numpy(regulationMatrix)
    if args.precisionModel == 'Double':
        regulationMatrix = regulationMatrix.type(torch.DoubleTensor)
    elif args.precisionModel == 'Float':
        regulationMatrix = regulationMatrix.type(torch.FloatTensor)
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+'---LTMG has been successfully prepared.')
else:
    regulationMatrix = None

# Original
if args.model == 'VAE':
    model = VAE(dim=scData.features.shape[1]).to(device)
elif args.model == 'AE':
    model = AE(dim=scData.features.shape[1]).to(device)
if args.precisionModel == 'Double':
    model = model.double()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))) +
      '---Pytorch model ready.')

def train(epoch, train_loader=train_loader, EMFlag=False, taskType='celltype', sparseImputation='nonsparse'):
    '''
    EMFlag indicates whether in EM processes. 
        If in EM, use regulized-type parsed from program entrance,
        Otherwise, noregu
        taskType: celltype or imputation
    '''
    model.train()
    train_loss = 0
    for batch_idx, (data, dataindex) in enumerate(train_loader):
        if args.precisionModel == 'Double':
            data = data.type(torch.DoubleTensor)
        elif args.precisionModel == 'Float':
            data = data.type(torch.FloatTensor)
        data = data.to(device)
        if not args.regulized_type == 'noregu':
            regulationMatrixBatch = regulationMatrix[dataindex, :]
            regulationMatrixBatch = regulationMatrixBatch.to(device)
        else:
            regulationMatrixBatch = None
        if taskType == 'imputation':
            if sparseImputation == 'nonsparse':
                celltypesampleBatch = celltypesample[dataindex,
                                                     :][:, dataindex]
                adjsampleBatch = adjsample[dataindex, :][:, dataindex]
            elif sparseImputation == 'sparse':
                celltypesampleBatch = generateCelltypeRegu(
                    listResult[dataindex])
                celltypesampleBatch = torch.from_numpy(celltypesampleBatch)
                if args.precisionModel == 'Float':
                    celltypesampleBatch = celltypesampleBatch.float()
                elif args.precisionModel == 'Double':
                    celltypesampleBatch = celltypesampleBatch.type(
                        torch.DoubleTensor)
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('celltype Mem consumption: '+str(mem))

                adjsampleBatch = adj[dataindex, :][:, dataindex]
                adjsampleBatch = sp.csr_matrix.todense(adjsampleBatch)
                adjsampleBatch = torch.from_numpy(adjsampleBatch)
                if args.precisionModel == 'Float':
                    adjsampleBatch = adjsampleBatch.float()
                elif args.precisionModel == 'Double':
                    adjsampleBatch = adjsampleBatch.type(torch.DoubleTensor)
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('adj Mem consumption: '+str(mem))

        optimizer.zero_grad()
        if args.model == 'VAE':
            recon_batch, mu, logvar, z = model(data)
            if taskType == 'celltype':
                if EMFlag and (not args.EMreguTag):
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch,
                                               regularizer_type='noregu', reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                else:
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch,
                                               regularizer_type=args.regulized_type, reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
            elif taskType == 'imputation':
                if EMFlag and (not args.EMreguTag):
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)
                else:
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)

        elif args.model == 'AE':
            recon_batch, z = model(data)
            mu_dummy = ''
            logvar_dummy = ''
            if taskType == 'celltype':
                if EMFlag and (not args.EMreguTag):
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=args.gammaPara,
                                               regulationMatrix=regulationMatrixBatch, regularizer_type='noregu', reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
                else:
                    loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch,
                                               regularizer_type=args.regulized_type, reguPara=args.alphaRegularizePara, modelusage=args.model, reduction=args.reduction)
            elif taskType == 'imputation':
                if EMFlag and (not args.EMreguTag):
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)
                else:
                    loss = loss_function_graph_celltype(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsampleBatch, celltyperegu=celltypesampleBatch, gammaPara=args.gammaImputePara,
                                                        regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.graphImputePara, reguParaCelltype=args.celltypeImputePara, modelusage=args.model, reduction=args.reduction)

        # L1 and L2 regularization
        # 0.0 for no regularization
        l1 = 0.0
        l2 = 0.0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
            l2 = l2 + p.pow(2).sum()
        loss = loss + args.L1Para * l1 + args.L2Para * l2

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
            recon_batch_all = recon_batch
            data_all = data
            z_all = z
        else:
            recon_batch_all = torch.cat((recon_batch_all, recon_batch), 0)
            data_all = torch.cat((data_all, data), 0)
            z_all = torch.cat((z_all, z), 0)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

    return recon_batch_all, data_all, z_all


if __name__ == "__main__":
    start_time = time.time()
    adjsample = None
    celltypesample = None
    # If not exist, then create the outputDir
    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    # outParaTag = str(args.gammaImputePara)+'-'+str(args.graphImputePara)+'-'+str(args.celltypeImputePara)
    ptfileStart = args.outputDir+args.datasetName+'_EMtrainingStart.pt'
    # ptfile      = args.outputDir+args.datasetName+'_EMtraining.pt'

    # Debug
    if args.debugMode == 'savePrune' or args.debugMode == 'noDebug':
        # store parameter
        stateStart = {
            # 'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(stateStart, ptfileStart)
        print('Start training...')
        for epoch in range(1, args.Regu_epochs + 1):
            recon, original, z = train(epoch, EMFlag=False)

        zOut = z.detach().cpu().numpy()
        print('zOut ready at ' + str(time.time()-start_time))
        ptstatus = model.state_dict()

        # Store reconOri for imputation
        reconOri = recon.clone()
        reconOri = reconOri.detach().cpu().numpy()

        # Step 1. Inferring celltype

        # Here para = 'euclidean:10'
        # adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k))
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time)))+'---Start Prune')
        adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(
            args.k), adjTag=(args.useGAEembedding or args.useBothembedding))
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Prune Finished')
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

        if args.debugMode == 'savePrune':
            # Add protocol=4 for serizalize object larger than 4GiB
            with open('edgeListFile', 'wb') as edgeListFile:
                pkl.dump(edgeList, edgeListFile, protocol=4)

            with open('adjFile', 'wb') as adjFile:
                pkl.dump(adj, adjFile, protocol=4)

            with open('zOutFile', 'wb') as zOutFile:
                pkl.dump(zOut, zOutFile, protocol=4)

            with open('reconFile', 'wb') as reconFile:
                pkl.dump(recon, reconFile, protocol=4)

            with open('originalFile', 'wb') as originalFile:
                pkl.dump(original, originalFile, protocol=4)

            sys.exit(0)

    if args.debugMode == 'loadPrune':

        with open('edgeListFile', 'rb') as edgeListFile:
            edgeList = pkl.load(edgeListFile)

        with open('adjFile', 'rb') as adjFile:
            adj = pkl.load(adjFile)

        with open('zOutFile', 'rb') as zOutFile:
            zOut = pkl.load(zOutFile)

        with open('reconFile', 'rb') as reconFile:
            recon = pkl.load(reconFile)

        with open('originalFile', 'rb') as originalFile:
            original = pkl.load(originalFile)

        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

    # Whether use GAE embedding
    if args.useGAEembedding or args.useBothembedding:
        zDiscret = zOut > np.mean(zOut, axis=0)
        zDiscret = 1.0*zDiscret
        if args.useGAEembedding:
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            zOut = GAEembedding(zDiscret, adj, args)
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+"---GAE embedding finished")
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

        elif args.useBothembedding:
            zEmbedding = GAEembedding(zDiscret, adj, args)
            zOut = np.concatenate((zOut, zEmbedding), axis=1)

    # For iteration studies
    G0 = nx.Graph()
    G0.add_weighted_edges_from(edgeList)
    nlG0 = nx.normalized_laplacian_matrix(G0)
    # set iteration criteria for converge
    adjOld = nlG0
    # set celltype criteria for converge
    listResultOld = [1 for i in range(zOut.shape[0])]

    # Fill the zeros before EM iteration
    # TODO: better implementation later, now we don't filling zeros for now
    if args.zerofillFlag:
        for nz_index in range(len(scData.nz_i)):
            # tmp = scipy.sparse.lil_matrix.todense(scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]])
            # tmp = np.asarray(tmp).reshape(-1)[0]
            tmp = scData.features[scData.nz_i[nz_index], scData.nz_j[nz_index]]
            reconOut[scData.nz_i[nz_index], scData.nz_j[nz_index]] = tmp
        recon = reconOut

    # Define resolution
    # Default: auto, otherwise use user defined resolution
    if args.resolution == 'auto':
        if zOut.shape[0] < 2000:
            resolution = 0.8
        else:
            resolution = 0.5
    else:
        resolution = float(args.resolution)

    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+"---EM process starts")

    for bigepoch in range(0, args.EM_iteration):
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Start %sth iteration.' % (bigepoch))

        # Now for both methods, we need do clustering, using clustering results to check converge
        # Clustering: Get clusters
        if args.clustering_method == 'Louvain':
            listResult, size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            print('Louvain cluster: '+str(k))
        elif args.clustering_method == 'LouvainK':
            listResult, size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            print('Louvain cluster: '+str(k))
            k = int(k*resolution) if int(k*resolution)>=3 else 2
            clustering = KMeans(n_clusters=k, random_state=0).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'LouvainB':
            listResult, size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            print('Louvain cluster: '+str(k))
            k = int(k*resolution) if int(k*resolution)>=3 else 2
            clustering = Birch(n_clusters=k).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'KMeans':
            clustering = KMeans(n_clusters=args.n_clusters,
                                random_state=0).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'SpectralClustering':
            clustering = SpectralClustering(
                n_clusters=args.n_clusters, assign_labels="discretize", random_state=0).fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method == 'AffinityPropagation':
            clustering = AffinityPropagation().fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'AgglomerativeClustering':
            clustering = AgglomerativeClustering().fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method == 'AgglomerativeClusteringK':
            clustering = AgglomerativeClustering(
                n_clusters=args.n_clusters).fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method == 'Birch':
            clustering = Birch(n_clusters=args.n_clusters).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'BirchN':
            clustering = Birch(n_clusters=None).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'MeanShift':
            clustering = MeanShift().fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method == 'OPTICS':
            clustering = OPTICS(min_samples=int(
                args.k/2), min_cluster_size=args.minMemberinCluster).fit(zOut)
            listResult = clustering.predict(zOut)
        else:
            print("Error: Clustering method not appropriate")
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+"---Clustering Ends")

        # If clusters more than maxclusters, then have to stop
        if len(set(listResult)) > args.maxClusterNumber or len(set(listResult)) <= 1:
            print("Stopping: Number of clusters is " +
                  str(len(set(listResult))) + ".")
            # Exit
            # return None
            # Else: dealing with the number
            listResult = trimClustering(
                listResult, minMemberinCluster=args.minMemberinCluster, maxClusterNumber=args.maxClusterNumber)

        # Debug: Calculate silhouette
        # measure_clustering_results(zOut, listResult)
        print('Total Cluster Number: '+str(len(set(listResult))))
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

        # Graph regulizated EM AE with Cluster AE, do the additional AE
        if not args.quickmode:
            # Each cluster has a autoencoder, and organize them back in iteraization
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Start Cluster Autoencoder.')
            clusterIndexList = []
            for i in range(len(set(listResult))):
                clusterIndexList.append([])
            for i in range(len(listResult)):
                assignee = listResult[i]
                # Avoid bugs for maxClusterNumber
                if assignee == args.maxClusterNumber:
                    assignee = args.maxClusterNumber-1
                clusterIndexList[assignee].append(i)

            reconNew = np.zeros(
                (scData.features.shape[0], scData.features.shape[1]))

            # Convert to Tensor
            reconNew = torch.from_numpy(reconNew)
            if args.precisionModel == 'Double':
                reconNew = reconNew.type(torch.DoubleTensor)
            elif args.precisionModel == 'Float':
                reconNew = reconNew.type(torch.FloatTensor)
            reconNew = reconNew.to(device)

            model.load_state_dict(ptstatus)

            for clusterIndex in clusterIndexList:
                reconUsage = recon[clusterIndex]
                scDataInter = scDatasetInter(reconUsage)
                train_loader = DataLoader(
                    scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)
                for epoch in range(1, args.cluster_epochs + 1):
                    reconCluster, originalCluster, zCluster = train(
                        epoch,  EMFlag=True)
                        # epoch, train_loader=train_loader, EMFlag=True)
                count = 0
                for i in clusterIndex:
                    reconNew[i] = reconCluster[count, :]
                    count += 1
                # empty cuda cache
                del originalCluster
                del zCluster
                torch.cuda.empty_cache()

            # Update
            recon = reconNew
            ptstatus = model.state_dict()

            # Debug mem consumption
            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))

        # Use new dataloader
        scDataInter = scDatasetInter(recon)
        train_loader = DataLoader(
            scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)

        for epoch in range(1, args.EM_epochs + 1):
            recon, original, z = train(epoch, EMFlag=True)
            # recon, original, z = train(epoch, train_loader=train_loader, EMFlag=True)

        zOut = z.detach().cpu().numpy()

        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time)))+'---Start Prune')
        adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para=args.knn_distance+':'+str(
            args.k), adjTag=(args.useGAEembedding or args.useBothembedding or (bigepoch == int(args.EM_iteration)-1)))
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Prune Finished')
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

        # Whether use GAE embedding
        if args.useGAEembedding or args.useBothembedding:
            zDiscret = zOut > np.mean(zOut, axis=0)
            zDiscret = 1.0*zDiscret
            if args.useGAEembedding:
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))
                zOut = GAEembedding(zDiscret, adj, args)
                print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                               start_time)))+"---GAE embedding finished")
                # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print('Mem consumption: '+str(mem))
            elif args.useBothembedding:
                zEmbedding = GAEembedding(zDiscret, adj, args)
                zOut = np.concatenate((zOut, zEmbedding), axis=1)

        # Original save step by step
        if args.saveinternal:
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Start save internal results')
            reconOut = recon.detach().cpu().numpy()

            # Output
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Prepare save')
            # print('Save results with reconstructed shape:'+str(reconOut.shape)+' Size of gene:'+str(len(genelist))+' Size of cell:'+str(len(celllist)))
            recon_df = pd.DataFrame(np.transpose(
                reconOut), index=genelist, columns=celllist)
            recon_df.to_csv(args.outputDir+args.datasetName+'_'+args.regulized_type+'_'+str(
                args.alphaRegularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_recon_'+str(bigepoch)+'.csv')
            emblist = []
            for i in range(zOut.shape[1]):
                emblist.append('embedding'+str(i))
            embedding_df = pd.DataFrame(zOut, index=celllist, columns=emblist)
            embedding_df.to_csv(args.outputDir+args.datasetName+'_'+args.regulized_type+'_'+str(
                args.alphaRegularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_embedding_'+str(bigepoch)+'.csv')
            graph_df = pd.DataFrame(
                edgeList, columns=["NodeA", "NodeB", "Weights"])
            graph_df.to_csv(args.outputDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.alphaRegularizePara) +
                            '_'+str(args.L1Para)+'_'+str(args.L2Para)+'_graph_'+str(bigepoch)+'.csv', index=False)
            results_df = pd.DataFrame(
                listResult, index=celllist, columns=["Celltype"])
            results_df.to_csv(args.outputDir+args.datasetName+'_'+args.regulized_type+'_'+str(
                args.alphaRegularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_results_'+str(bigepoch)+'.txt')

            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---Save internal completed')

        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))
        print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                       start_time)))+'---Start test converge condition')

        # Iteration usage
        # If not only use 'celltype', we have to use graph change
        # The problem is it will consume huge memory for giant graphs
        if not args.converge_type == 'celltype':
            Gc = nx.Graph()
            Gc.add_weighted_edges_from(edgeList)
            adjGc = nx.adjacency_matrix(Gc)

            # Update new adj
            adjNew = args.alpha*nlG0 + \
                (1-args.alpha) * adjGc/np.sum(adjGc, axis=0)

            # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print('Mem consumption: '+str(mem))
            print('---'+str(datetime.timedelta(seconds=int(time.time() -
                                                           start_time)))+'---New adj ready')

            # debug
            graphChange = np.mean(abs(adjNew-adjOld))
            graphChangeThreshold = args.converge_graphratio * \
                np.mean(abs(nlG0))
            print('adjNew:{} adjOld:{} G0:{}'.format(adjNew, adjOld, nlG0))
            print('mean:{} threshold:{}'.format(
                graphChange, graphChangeThreshold))

            # Update
            adjOld = adjNew

        # Check similarity
        ari = adjusted_rand_score(listResultOld, listResult)

        # Debug Information of clustering results between iterations
        # print(listResultOld)
        # print(listResult)
        print('celltype similarity:'+str(ari))

        # graph criteria
        if args.converge_type == 'graph':
            if graphChange < graphChangeThreshold:
                print('Converge now!')
                break
        # celltype criteria
        elif args.converge_type == 'celltype':
            if ari > args.converge_celltyperatio:
                print('Converge now!')
                break
        # if both criteria are meets
        elif args.converge_type == 'both':
            if graphChange < graphChangeThreshold and ari > args.converge_celltyperatio:
                print('Converge now!')
                break
        # if either criteria are meets
        elif args.converge_type == 'either':
            if graphChange < graphChangeThreshold or ari > args.converge_celltyperatio:
                print('Converge now!')
                break

        # Update
        listResultOld = listResult
        print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                        )+"---"+str(bigepoch)+"th iteration in EM Finished")

    # Use new dataloader
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+"---Starts Imputation")
    # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # print('Mem consumption: '+str(mem))
    scDataInter = scDatasetInter(reconOri)
    train_loader = DataLoader(
        scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)

    stateStart = torch.load(ptfileStart)
    model.load_state_dict(stateStart['state_dict'])
    optimizer.load_state_dict(stateStart['optimizer'])
    # model.load_state_dict(torch.load(ptfileStart))
    # if args.aePara == 'start':
    #     model.load_state_dict(torch.load(ptfileStart))
    # elif args.aePara == 'end':
    #     model.load_state_dict(torch.load(ptfileEnd))

    # generate graph regularizer from graph
    # adj = adj.tolist() # Used for read/load
    # adjdense = sp.csr_matrix.todense(adj)

    # Better option: use torch.sparse
    if args.sparseImputation == 'nonsparse':
        # generate adj from edgeList
        adjdense = sp.csr_matrix.todense(adj)
        adjsample = torch.from_numpy(adjdense)
        if args.precisionModel == 'Float':
            adjsample = adjsample.float()
        elif args.precisionModel == 'Double':
            adjsample = adjsample.type(torch.DoubleTensor)
        adjsample = adjsample.to(device)
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

        # generate celltype regularizer from celltype
        celltypesample = generateCelltypeRegu(listResult)
        celltypesample = torch.from_numpy(celltypesample)
        if args.precisionModel == 'Float':
            celltypesample = celltypesample.float()
        elif args.precisionModel == 'Double':
            celltypesample = celltypesample.type(torch.DoubleTensor)
        celltypesample = celltypesample.to(device)
        # mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption: '+str(mem))

    for epoch in range(1, args.EM_epochs + 1):
        recon, original, z = train(
            epoch, EMFlag=True, taskType='imputation', sparseImputation=args.sparseImputation)

    reconOut = recon.detach().cpu().numpy()
    if not args.noPostprocessingTag:
        threshold_indices = reconOut < args.postThreshold
        reconOut[threshold_indices] = 0.0

    # Output final results
    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+'---All iterations finished, start output results.')
    # Output imputation Results
    # np.save   (args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+outParaTag+'_recon.npy',reconOut)
    # np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+outParaTag+'_recon.csv',reconOut,delimiter=",",fmt='%10.4f')
    # Output celltype Results
    recon_df = pd.DataFrame(np.transpose(reconOut),
                            index=genelist, columns=celllist)
    recon_df.to_csv(args.outputDir+args.datasetName+'_recon.csv')
    emblist = []
    for i in range(zOut.shape[1]):
        emblist.append('embedding'+str(i))
    embedding_df = pd.DataFrame(zOut, index=celllist, columns=emblist)
    embedding_df.to_csv(args.outputDir+args.datasetName+'_embedding.csv')
    graph_df = pd.DataFrame(edgeList, columns=["NodeA", "NodeB", "Weights"])
    graph_df.to_csv(args.outputDir+args.datasetName+'_graph.csv', index=False)
    results_df = pd.DataFrame(listResult, index=celllist, columns=["Celltype"])
    results_df.to_csv(args.outputDir+args.datasetName+'_results.txt')

    print('---'+str(datetime.timedelta(seconds=int(time.time()-start_time))
                    )+"---scGNN finished")
