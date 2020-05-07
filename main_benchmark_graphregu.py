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
from LTMG_R import *
import pandas as pd
import torch.multiprocessing as mp

# Used only for benchmark, needs Preprocessing_main.py first, then proceed by this script.
parser = argparse.ArgumentParser(description='Graph EM AutoEncoder for scRNA')
parser.add_argument('--datasetName', type=str, default='1.Biase',
                    help='TGFb/sci-CAR/sci-CAR_LTMG/MMPbasal/MMPbasal_all/MMPbasal_allgene/MMPbasal_allcell/MMPepo/MMPbasal_LTMG/MMPbasal_all_LTMG/MMPbasal_2000')
# Dataset: 1-13 benchmark: 1.Biase/2.Li/3.Treutlein/4.Yan/5.Goolam/6.Guo/7.Deng/8.Pollen/9.Chung/10.Usoskin/11.Kolodziejczyk/12.Klein/13.Zeisel
parser.add_argument('--batch-size', type=int, default=12800, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--EM-epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train in iteration EM (default: 50)')
parser.add_argument('--EM-iteration', type=int, default=10, metavar='N',
                    help='number of epochs in EM iteration (default: 10)')
parser.add_argument('--EMtype', type=str, default='EM',
                    help='EM process type (default: celltypeEM) or EM')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='iteration alpha (default: 0.5) to control the converge rate, should be a number between 0~1')
parser.add_argument('--converge-type', type=str, default='either',
                    help='type of converge: celltype/graph/both/either (default: celltype) ')
parser.add_argument('--converge-graphratio', type=float, default=0.01,
                    help='ratio of cell type change in EM iteration (default: 0.01), 0-1')
parser.add_argument('--converge-celltyperatio', type=float, default=0.99,
                    help='ratio of cell type change in EM iteration (default: 0.99), 0-1')
parser.add_argument('--celltype-epochs', type=int, default=50, metavar='N',
                    help='number of epochs in celltype training (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--regulized-type', type=str, default='LTMG',
                    help='regulized type (default: Graph) in EM, otherwise: noregu/LTMG/LTMG01')
parser.add_argument('--EMregulized-type', type=str, default='Graph',
                    help='regulized type (default: noregu) in EM, otherwise: noregu/Graph/GraphR') 
parser.add_argument('--ONEregulized-type', type=str, default='NA', 
                    help='regulized type (default: NA) in oneImpute, otherwise: NA/LTMG-Graph/LTMG-GraphR')
parser.add_argument('--adjtype', type=str, default='weighted',
                    help='adjtype (default: weighted) otherwise: unweighted') 
                   
parser.add_argument('--gammaPara', type=float, default=0.1,
                    help='regulized parameter (default: 1.0)')
parser.add_argument('--regularizePara', type=float, default=0.9,
                    help='regulized parameter (default: 0.001)')
parser.add_argument('--L1Para', type=float, default=0.0,
                    help='regulized parameter (default: 0.001)')
parser.add_argument('--L2Para', type=float, default=0.0,
                    help='regulized parameter (default: 0.001)')
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
parser.add_argument('--inferLTMGTag', action='store_true', default=False,
                    help='Whether infer LTMG')                   
parser.add_argument('--LTMGDir', type=str, default='/home/jwang/data/scData/',
                    help='directory of LTMGDir, default:(/home/wangjue/biodata/scData/allBench/)')
parser.add_argument('--expressionFile', type=str, default='Biase_expression.csv',
                    help='expression File in csv')
parser.add_argument('--ltmgFile', type=str, default='ltmg.csv',
                    help='expression File in csv')

#Clustering related
parser.add_argument('--useGAEembedding', action='store_true', default=False, 
                    help='whether use GAE embedding for clustering(default: False)')
parser.add_argument('--useBothembedding', action='store_true', default=False, 
                    help='whether use both embedding and Graph embedding for clustering(default: False)')
parser.add_argument('--clustering-method', type=str, default='LouvainK',
                    help='Clustering method: Louvain/KMeans/SpectralClustering/AffinityPropagation/AgglomerativeClustering/Birch/BirchN/MeanShift/OPTICS/LouvainK/LouvainB')
parser.add_argument('--maxClusterNumber', type=int, default=30,
                    help='max cluster for celltypeEM without setting number of clusters (default: 30)') 
parser.add_argument('--minMemberinCluster', type=int, default=5,
                    help='max cluster for celltypeEM without setting number of clusters (default: 100)')
parser.add_argument('--resolution', type=str, default='auto',
                    help='the number of resolution on Louvain (default: auto/0.5/0.8)')
parser.add_argument('--prunetype', type=str, default='KNNgraphStatsSingleThread',
                    help='prune type, KNNgraphStats/KNNgraphML/KNNgraphStatsSingleThread (default: KNNgraphStats)')
parser.add_argument('--EMreguTag', action='store_true', default=False,
                    help='whether regu in EM process')

#Benchmark related
parser.add_argument('--benchmark', type=str, default='/home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv',
                    help='the benchmark file of celltype (default: /home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv)')

#Aggrelated
parser.add_argument('--linkage', type=str, default='ward',
                    help='linkage should be: ward, average, complete, single')

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
    # if args.discreteTag:
    #     scData = scBenchDataset(args.datasetName, args.discreteTag)
    # else:
    #     scData = scBenchDataset(args.datasetName, args.discreteTag, transform=logtransform)
    scData = scBenchDataset(args.datasetName, args.discreteTag)
else:
    # if args.discreteTag:
    #     scData = scDatasetDropout(args.datasetName, args.discreteTag, args.dropoutRatio)
    # else:
    #     scData = scDatasetDropout(args.datasetName, args.discreteTag, args.dropoutRatio, transform=logtransform)
    scData = scDatasetDropout(datasetName=args.datasetName, discreteTag=args.discreteTag, ratio=args.dropoutRatio, seed=args.seed)
train_loader = DataLoader(scData, batch_size=args.batch_size, shuffle=False, **kwargs)

if args.inferLTMGTag:
    #run LTMG in R
    runLTMG(args.LTMGDir+'test/'+args.expressionFile,args.LTMGDir+'test/')
    ltmgFile = args.ltmgFile
else:
    ltmgFile = args.datasetName+'/T2000_UsingOriginalMatrix/T2000_LTMG.txt'

regulationMatrix = readLTMGnonsparse(args.LTMGDir, ltmgFile)
regulationMatrix = torch.from_numpy(regulationMatrix)

# Original
if args.model == 'VAE':
    # model = VAE(dim=scData.features.shape[1]).to(device)
    model = VAE2d(dim=scData.features.shape[1]).to(device)
elif args.model == 'AE':
    model = AE(dim=scData.features.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#Benchmark
bench_pd=pd.read_csv(args.benchmark,index_col=0)
#t1=pd.read_csv('/home/jwang/data/scData/13.Zeisel/Zeisel_cell_label.csv',index_col=0)
bench_celltype=bench_pd.iloc[:,0].to_numpy()

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
        # print('))'+str(batch_idx))
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        # print(')))'+str(data.shape))
        regulationMatrixBatch = regulationMatrix[dataindex,:]
        # print('))))'+str(regulationMatrixBatch.shape))
        optimizer.zero_grad()
        # print('**afterzero')
        if args.model == 'VAE':
            recon_batch, mu, logvar, z = model(data)
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            if EMFlag and (not args.EMreguTag):
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsample, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.regularizePara, modelusage=args.model)
            else: 
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsample, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.regularizePara, modelusage=args.model)    
            
        elif args.model == 'AE':
            recon_batch, z = model(data)
            mu_dummy = ''
            logvar_dummy = ''
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            if EMFlag and (not args.EMreguTag):
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsample, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch, regularizer_type=args.EMregulized_type, reguPara=args.regularizePara, modelusage=args.model)    
            else:
                loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsample, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch, regularizer_type=args.regulized_type, reguPara=args.regularizePara, modelusage=args.model)
               
        # L1 and L2 regularization
        # 0.0 for no regularization 
        l1 = 0.0
        l2 = 0.0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
            l2 = l2 + p.pow(2).sum()
        loss = loss + args.L1Para * l1 + args.L2Para * l2
        # print('**preback')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print('((afterback')
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

def trainONE(epoch, model = model, train_loader=train_loader, EMFlag=False):
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
        # print('))'+str(batch_idx))
        data = data.type(torch.FloatTensor)
        data = data.to(device)
        # print(')))'+str(data.shape))
        regulationMatrixBatch = regulationMatrix[dataindex,:]
        # print('))))'+str(regulationMatrixBatch.shape))
        optimizer.zero_grad()
        # print('**afterzero')
        if args.model == 'VAE':
            recon_batch, mu, logvar, z = model(data)
            # Original
            loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu, logvar, graphregu=adjsample, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch, regularizer_type=args.ONEregulized_type, reguPara=args.regularizePara, modelusage=args.model)    
            
        elif args.model == 'AE':
            recon_batch, z = model(data)
            mu_dummy = ''
            logvar_dummy = ''
            # Original
            # loss = loss_function(recon_batch, data, mu, logvar)
            loss = loss_function_graph(recon_batch, data.view(-1, recon_batch.shape[1]), mu_dummy, logvar_dummy, graphregu=adjsample, gammaPara=args.gammaPara, regulationMatrix=regulationMatrixBatch, regularizer_type=args.ONEregulized_type, reguPara=args.regularizePara, modelusage=args.model)
               
        # L1 and L2 regularization
        # 0.0 for no regularization 
        l1 = 0.0
        l2 = 0.0
        for p in model.parameters():
            l1 = l1 + p.abs().sum()
            l2 = l2 + p.pow(2).sum()
        loss = loss + args.L1Para * l1 + args.L2Para * l2
        # print('**preback')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        # print('((afterback')
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

def trainParallel(i,recon,clusterIndexList,args):
        '''
        Train each autoencoder in paral
        '''
        print('~'+str(i))
        clusterIndex = clusterIndexList[i]
        print('!'+str(clusterIndex))
        reconUsage = recon[clusterIndex]
        print('@'+str(reconUsage))
        scDataInter = scDatasetInter(reconUsage)
        train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)
        for epoch in range(1, args.celltype_epochs + 1):
            print('#'+str(epoch))
            reconCluster, originalCluster, zCluster = train(epoch, EMFlag=True) 
        print('$'+str(i))
        count = 0
        for i in clusterIndex:
            reconNew[i] = reconCluster[count,:]
            count +=1

if __name__ == "__main__":
    adjsample = None
    ptfile = args.npyDir+args.datasetName+'_'+str(args.regularizePara)+'_EMtraining.pt'
    start_time = time.time()
    discreteStr = ''
    if args.discreteTag:
        discreteStr = 'D'

    # Save results only when impute
    if args.imputeMode:
        # Does not need now
        # save_sparse_matrix(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
        # sp.save_npz(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_features.npz',scData.features)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_features.npy',scData.features)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_dropi.npy',scData.i)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_dropj.npy',scData.j)
        np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_dropix.npy',scData.ix)

    for epoch in range(1, args.epochs + 1):
        recon, original, z = train(epoch, EMFlag=False)
        
    zOut = z.detach().cpu().numpy() 
    torch.save(model.state_dict(),ptfile)

    #Define resolution
    #Default: auto, otherwise use user defined resolution
    if args.resolution == 'auto':
        if zOut.shape[0]< 2000:
            resolution = 0.8
        else:
            resolution = 0.5
    else:
        resolution = float(args.resolution)

    prune_time = time.time()        
    # Here para = 'euclidean:10'
    # adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
    if adjtype=='unweighted':
        adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para = args.knn_distance+':'+str(args.k), outAdjTag = (args.useGAEembedding or args.useBothembedding))
        adjdense = sp.csr_matrix.todense(adj)
    elif adjtype=='weighted':
        adj, edgeList = generateAdjWeighted(zOut, graphType=args.prunetype, para = args.knn_distance+':'+str(args.k), outAdjTag = (args.useGAEembedding or args.useBothembedding))   
        adjdense = adj.todense(adj)
    adjsample = torch.from_numpy(adjdense)
    adjsample = adjsample.float()
    print("---Pruning takes %s seconds ---" % (time.time() - prune_time))
    if args.saveFlag:
        reconOut = recon.detach().cpu().numpy()
        if args.imputeMode:
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_recon.npy',reconOut)
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_z.npy',zOut)
        else:  
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.regularizePara)+'_recon.npy',reconOut)
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.regularizePara)+'_z.npy',zOut)
    
    # Whether use GAE embedding
    if args.useGAEembedding or args.useBothembedding:
        zDiscret = zOut>np.mean(zOut,axis=0)
        zDiscret = 1.0*zDiscret
        if args.useGAEembedding:
            zOut=GAEembedding(zDiscret, adj, args)
        elif args.useBothembedding:
            zEmbedding=GAEembedding(zDiscret, adj, args)
            zOut=np.concatenate((zOut,zEmbedding),axis=1)
        # Debug!!! No use Embedding
        #
        # prune_time = time.time()
        # # Here para = 'euclidean:10'
        # adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
        # adjdense = sp.csr_matrix.todense(adj)
        # adjsample = torch.from_numpy(adjdense)
        # print("---Pruning takes %s seconds ---" % (time.time() - prune_time))
        # if args.saveFlag:
        #     if args.imputeMode:
        #         np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_zGAE.npy',zOut)
        #     else:
        #         np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_zGAE.npy',zOut)
        # # np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_init_edgeList.npy',edgeList)
    
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
            k = len(np.unique(listResult))
            print('Louvain cluster: '+str(k))
        elif args.clustering_method=='LouvainK':
            from R_util import generateLouvainCluster
            listResult,size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            print('Louvain cluster: '+str(k)+' with resolution'+str(resolution))
            # resolution of louvain cluster:
            k = int(k*resolution) if k>3 else 2
            clustering = KMeans(n_clusters=k, random_state=0).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method=='LouvainB':
            from R_util import generateLouvainCluster
            listResult,size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            print('Louvain cluster: '+str(k))
            # resolution of louvain cluster:
            k = int(k*resolution) if k>3 else 2
            clustering = Birch(n_clusters=k).fit(zOut)
            listResult = clustering.predict(zOut)
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
            clustering = AgglomerativeClustering(linkage=args.linkage).fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method=='Birch':
            clustering = Birch(n_clusters=args.n_clusters).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method=='BirchN':
            clustering = Birch(n_clusters=None).fit(zOut)
            listResult = clustering.predict(zOut)
        elif args.clustering_method=='MeanShift':
            clustering = MeanShift().fit(zOut)
            listResult = clustering.labels_.tolist()
        elif args.clustering_method=='OPTICS':
            clustering = OPTICS(min_samples=int(args.k/2), min_cluster_size=args.minMemberinCluster).fit(zOut)
            listResult = clustering.labels_.tolist()
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

            # No parallel
            for clusterIndex in clusterIndexList:
                model.load_state_dict(torch.load(ptfile))
                reconUsage = recon[clusterIndex]
                scDataInter = scDatasetInter(reconUsage)
                train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)
                for epoch in range(1, args.celltype_epochs + 1):
                    reconCluster, originalCluster, zCluster = train(epoch, EMFlag=True)                
                count = 0
                for i in clusterIndex:
                    reconNew[i] = reconCluster[count,:]
                    count +=1
            
            # parallel
            # TODO: looks like pytorch does not support this
            # reconNew.share_memory_()
            # celltypeNum = len(clusterIndexList)
            # processes=[]

            # for i in range(celltypeNum):
            #     #call Function: trainParallel(i,recon,clusterIndexList,args)
            #     px = mp.Process(target=trainParallel, args=(i,recon,clusterIndexList,args))
            #     px.start()
            #     processes.append(px)
            # for px in processes:
            #     px.join()
            #     # foo(i,tl)
            
            # Update
            recon = reconNew
        
        # Use new dataloader
        scDataInter = scDatasetInter(recon)
        train_loader = DataLoader(scDataInter, batch_size=args.batch_size, shuffle=False, **kwargs)

        model.load_state_dict(torch.load(ptfile))
        for epoch in range(1, args.EM_epochs + 1):
            recon, original, z = train(epoch, EMFlag=True)
        
        zOut = z.detach().cpu().numpy()
        torch.save(model.state_dict(),ptfile)

        prune_time = time.time()
        # Here para = 'euclidean:10'
        # adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k))
        if adjtype == 'unweighted':
            adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para = args.knn_distance+':'+str(args.k), outAdjTag = (args.useGAEembedding or args.useBothembedding)) 
            adjdense = sp.csr_matrix.todense(adj)
        elif adjtype == 'weighted':
            adj, edgeList = generateAdjWeighted(zOut, graphType=args.prunetype, para = args.knn_distance+':'+str(args.k), outAdjTag = (args.useGAEembedding or args.useBothembedding))         
            adjdense = adj.todense(adj)
        adjsample = torch.from_numpy(adjdense)
        adjsample = adjsample.float()
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
            # Debug!!
            # prune_time = time.time()
            # # Here para = 'euclidean:10'
            # adj, edgeList = generateAdj(zOut, graphType='KNNgraphML', para = args.knn_distance+':'+str(args.k)) 
            # adjdense = sp.csr_matrix.todense(adj)
            # adjsample = torch.from_numpy(adjdense)
            # print("---Pruning takes %s seconds ---" % (time.time() - prune_time))

        if args.saveFlag:
            reconOut = recon.detach().cpu().numpy()
            if args.imputeMode:
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_recon'+str(bigepoch)+'.npy',reconOut)
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_z'+str(bigepoch)+'.npy',zOut)
            else:
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.regularizePara)+'_recon'+str(bigepoch)+'.npy',reconOut)
                np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.regularizePara)+'_z'+str(bigepoch)+'.npy',zOut)
        
        print("---One iteration in EM process, proceeded %s seconds ---" % (time.time() - iteration_time))

        #Iteration usage
        Gc = nx.Graph()
        Gc.add_weighted_edges_from(edgeList)
        adjGc = nx.adjacency_matrix(Gc)
        
        # Update new adj
        adjNew = args.alpha*nlG0 + (1-args.alpha) * adjGc/np.sum(adjGc,axis=0)
        
        #debug
        graphChange = np.mean(abs(adjNew-adjOld))
        graphChangeThreshold = args.converge_graphratio * np.mean(abs(nlG0))
        print('adjNew:{} adjOld:{} G0:{}'.format(adjNew, adjOld, nlG0))
        print('mean:{} threshold:{}'.format(graphChange, graphChangeThreshold))
        silhouette, chs, dbs = measureClusteringNoLabel(zOut, listResult)
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(listResultOld, listResult)
        print(listResultOld)
        print(listResult)
        print('celltype similarity:'+str(ari))
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(bench_celltype, listResult)
        
        resultarray=[]
        resultstr = str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs)
        resultarray.append(resultstr)
        print('All Results: ')
        print(resultstr)

        if args.saveFlag:
            if args.imputeMode:
                np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_benchmark'+str(bigepoch)+'.txt',resultarray,fmt='%s')
            else:
                np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_benchmark'+str(bigepoch)+'.txt',resultarray,fmt='%s')


        # graph criteria
        if args.converge_type == 'graph':       
            if graphChange < graphChangeThreshold:
                print('Converge now!')
                break
        # celltype criteria
        elif args.converge_type == 'celltype':            
            if ari>args.converge_celltyperatio:
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
        adjOld = adjNew
        listResultOld = listResult
        # torch.cuda.empty_cache()
            
    if args.saveFlag:
        if args.imputeMode:
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_final_edgeList.npy',edgeList)
        else:
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.regularizePara)+'_final_edgeList.npy',edgeList)
        
        # recon_df = pd.DataFrame(reconOut,columns=genelist,index=celllist)
        # recon_df.to_csv(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_recon.csv')
        # embedding_df = pd.DataFrame(zOut,index=celllist)
        # embedding_df.to_csv(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_embedding.csv')    
        np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_recon.csv',reconOut,delimiter=",",fmt='%10.4f')
        np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_embedding.csv',zOut, delimiter=",",fmt='%10.4f')
        np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_graph.csv',edgeList,fmt='%d,%d,%2.1f')
        np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_results.txt',listResult,fmt='%d')

        resultarray=[]
        silhouette, chs, dbs = measureClusteringNoLabel(zOut, listResult)
        ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(bench_celltype, listResult)
        resultstr = str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs)
        resultarray.append(resultstr)
        np.savetxt(args.npyDir+args.datasetName+'_'+args.regulized_type+'_'+str(args.regularizePara)+'_'+str(args.L1Para)+'_'+str(args.L2Para)+'_benchmark.txt',resultarray,fmt='%s')

        if not args.ONEregulized_type == 'NA':
            if not args.imputeMode:
                scData = scBenchDataset(args.datasetName, args.discreteTag)
            else:
                scData = scDatasetDropout(datasetName=args.datasetName, discreteTag=args.discreteTag, ratio=args.dropoutRatio, seed=args.seed)
            train_loader = DataLoader(scData, batch_size=args.batch_size, shuffle=False, **kwargs)

            if args.inferLTMGTag:
                #run LTMG in R
                runLTMG(args.LTMGDir+'test/'+args.expressionFile,args.LTMGDir+'test/')
                ltmgFile = args.ltmgFile
            else:
                ltmgFile = args.datasetName+'/T2000_UsingOriginalMatrix/T2000_LTMG.txt'

            regulationMatrix = readLTMGnonsparse(args.LTMGDir, ltmgFile)
            regulationMatrix = torch.from_numpy(regulationMatrix)

            # Original
            if args.model == 'VAE':
                # model = VAE(dim=scData.features.shape[1]).to(device)
                model1 = VAE2d(dim=scData.features.shape[1]).to(device)
            elif args.model == 'AE':
                model1 = AE(dim=scData.features.shape[1]).to(device)
            optimizer = optim.Adam(model1.parameters(), lr=1e-3)

            for epoch in range(1, args.epochs + 1):
                recon, original, z = trainONE(epoch, model=model1, EMFlag=False)
                
            zOut = z.detach().cpu().numpy() 

            adj, edgeList = generateAdj(zOut, graphType=args.prunetype, para = args.knn_distance+':'+str(args.k), outAdjTag = (args.useGAEembedding or args.useBothembedding)) 

            #Define resolution
            #Default: auto, otherwise use user defined resolution
            if args.resolution == 'auto':
                if zOut.shape[0]< 2000:
                    resolution = 0.8
                else:
                    resolution = 0.5
            else:
                resolution = float(args.resolution)

            from R_util import generateLouvainCluster
            listResult,size = generateLouvainCluster(edgeList)
            k = len(np.unique(listResult))
            print('Louvain cluster: '+str(k))
            # resolution of louvain cluster:
            k = int(k*resolution) if k>3 else 2
            clustering = KMeans(n_clusters=k, random_state=0).fit(zOut)
            listResult = clustering.predict(zOut)

            ari, ami, nmi, cs, fms, vms, hs = measureClusteringTrueLabel(bench_celltype, listResult)
            print('All Results: ')
            print(str(silhouette)+' '+str(chs)+' '+str(dbs)+' '+str(ari)+' '+str(ami)+' '+str(nmi)+' '+str(cs)+' '+str(fms)+' '+str(vms)+' '+str(hs))
       
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_recon'+str(bigepoch+1)+'.npy',reconOut)
            np.save(args.npyDir+args.datasetName+'_'+args.regulized_type+discreteStr+'_'+str(args.dropoutRatio)+'_'+str(args.regularizePara)+'_z'+str(bigepoch+1)+'.npy',zOut)
 
    print("---Total Running Time: %s seconds ---" % (time.time() - start_time))
