import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
import argparse
import time
import random
import numpy as np
import scipy.sparse as sp
import torch
np.random.seed(SEED)
torch.manual_seed(SEED)
from torch import optim
import torch.nn.functional as F
from gae.model import GCNModelVAE, GCNModelAE
from gae.optimizer import loss_function
from gae.utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from tqdm import tqdm
from graph_function import *
from benchmark_util import *
import resource

# Ref codes from https://github.com/MysteryVaibhav/RWR-GAE
def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--npyDir',type=str,default='npyGraph10/',help="npyDir")
    parser.add_argument('--zFilename',type=str,default='5.Pollen_all_noregu_recon0.npy',help="z Filename")
    parser.add_argument('--benchmark',type=bool,default=True,help="whether have benchmark")
    # cell File
    parser.add_argument('--labelFilename',type=str,default='/home/wangjue/biodata/scData/AnjunBenchmark/5.Pollen/Pollen_cell_label.csv',help="label Filename")
    parser.add_argument('--originalFile',type=str,default='data/sc/5.Pollen_all/5.Pollen_all.features.csv',help="original csv Filename")
    # if use only part of the cells
    parser.add_argument('--cellFilename',type=str,default='/home/wangjue/biodata/scData/5.Pollen.cellname.txt',help="cell Filename")
    parser.add_argument('--cellIndexname',type=str,default='/home/wangjue/myprojects/scGNN/data/sc/5.Pollen_all/ind.5.Pollen_all.cellindex.txt',help="cell index Filename")

    # GAE
    parser.add_argument('--GAEmodel', type=str, default='gcn_vae', help="models used")
    parser.add_argument('--dw', type=int, default=0, help="whether to use deepWalk regularization, 0/1")
    parser.add_argument('--GAEepochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--GAEhidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--GAEhidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--GAElr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--GAEdropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
    parser.add_argument('--walk-length', default=5, type=int, help='Length of the random walk started at each node')
    parser.add_argument('--window-size', default=3, type=int, help='Window size of skipgram model.')
    parser.add_argument('--number-walks', default=5, type=int, help='Number of random walks to start at each node')
    parser.add_argument('--full-number-walks', default=0, type=int, help='Number of random walks from each node')
    parser.add_argument('--GAElr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
    parser.add_argument('--context', type=int, default=0, help="whether to use context nodes for skipgram")
    parser.add_argument('--ns', type=int, default=1, help="whether to use negative samples for skipgram")
    parser.add_argument('--n-clusters', default=11, type=int, help='number of clusters, 7 for cora, 6 for citeseer')
    parser.add_argument('--GAEplot', type=int, default=0, help="whether to plot the clusters using tsne")
    parser.add_argument('--precisionModel', type=str, default='Float', 
                    help='Single Precision/Double precision: Float/Double (default:Float)')
    args = parser.parse_args()

#gae embedding
def GAEembedding(z, adj, args):
    '''
    GAE embedding for clustering
    Param:
        z,adj
    Return:
        Embedding from graph
    '''   
    # featrues from z
    # Louvain
    features = z
    # features = torch.DoubleTensor(features)
    features = torch.FloatTensor(features)

    # Old implementation
    # adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
    
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    # adj_label = torch.DoubleTensor(adj_label.toarray())
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    if args.GAEmodel == 'gcn_vae':
        model = GCNModelVAE(feat_dim, args.GAEhidden1, args.GAEhidden2, args.GAEdropout)
    else:
        model = GCNModelAE(feat_dim, args.GAEhidden1, args.GAEhidden2, args.GAEdropout)
    if args.precisionModel == 'Double':
        model=model.double()
    optimizer = optim.Adam(model.parameters(), lr=args.GAElr)

    hidden_emb = None
    for epoch in tqdm(range(args.GAEepochs)):
        t = time.time()
        # mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print('Mem consumption before training: '+str(mem))
        model.train()
        optimizer.zero_grad()
        z, mu, logvar = model(features, adj_norm)

        loss = loss_function(preds=model.dc(z), labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        # TODO, this is prediction 
        # roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        ap_curr = 0

        tqdm.write("Epoch: {}, train_loss_gae={:.5f}, val_ap={:.5f}, time={:.5f}".format(
            epoch + 1, cur_loss,
            ap_curr, time.time() - t))


    tqdm.write("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    tqdm.write('Test ROC score: ' + str(roc_score))
    tqdm.write('Test AP score: ' + str(ap_score))

    return hidden_emb

if __name__=='__main__':
    main()

