from __future__ import division
from __future__ import print_function
import os, sys

# Main entrance from https://github.com/MysteryVaibhav/RWR-GAE
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
from model import GCNModelVAE, GCNModelAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from deepWalk.graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter, build_deepwalk_corpus
from deepWalk.skipGram import SkipGram
from sklearn.cluster import KMeans
from clustering_metric import clustering_metrics
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--dw', type=int, default=1, help="whether to use deepWalk regularization, 0/1")
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
parser.add_argument('--walk-length', default=5, type=int, help='Length of the random walk started at each node')
parser.add_argument('--window-size', default=3, type=int, help='Window size of skipgram model.')
parser.add_argument('--number-walks', default=5, type=int, help='Number of random walks to start at each node')
parser.add_argument('--full-number-walks', default=0, type=int, help='Number of random walks from each node')
parser.add_argument('--lr_dw', type=float, default=0.001, help='Initial learning rate for regularization.')
parser.add_argument('--context', type=int, default=0, help="whether to use context nodes for skipgram")
parser.add_argument('--ns', type=int, default=1, help="whether to use negative samples for skipgram")
parser.add_argument('--n-clusters', default=7, type=int, help='number of clusters, 7 for cora, 6 for citeseer')
parser.add_argument('--plot', type=int, default=0, help="whether to plot the clusters using tsne")
parser.add_argument('--precisionModel', type=str, default='Float', 
                    help='Single Precision/Double precision: Float/Double (default:Float)')
args = parser.parse_args()


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, y_test, tx, ty, test_maks, true_labels = load_data(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Before proceeding further, make the structure for doing deepWalk
    if args.dw == 1:
        print('Using deepWalk regularization...')
        G = load_edgelist_from_csr_matrix(adj_orig, undirected=True)
        print("Number of nodes: {}".format(len(G.nodes())))
        num_walks = len(G.nodes()) * args.number_walks
        print("Number of walks: {}".format(num_walks))
        data_size = num_walks * args.walk_length
        print("Data size (walks*length): {}".format(data_size))

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    # adj_label = torch.DoubleTensor(adj_label.toarray())
    adj_label = torch.FloatTensor(adj_label.toarray())

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    if args.model == 'gcn_vae':
        model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    else:
        model = GCNModelAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.dw == 1:
        sg = SkipGram(args.hidden2, adj.shape[0])
        optimizer_dw = optim.Adam(sg.parameters(), lr=args.lr_dw)

        # Construct the nodes for doing random walk. Doing it before since the seed is fixed
        nodes_in_G = list(G.nodes())
        chunks = len(nodes_in_G) // args.number_walks
        random.Random().shuffle(nodes_in_G)

    hidden_emb = None
    for epoch in tqdm(range(args.epochs)):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        z, mu, logvar = model(features, adj_norm)

        # After back-propagating gae loss, now do the deepWalk regularization
        if args.dw == 1:
            sg.train()
            if args.full_number_walks > 0:
                walks = build_deepwalk_corpus(G, num_paths=args.full_number_walks,
                                              path_length=args.walk_length, alpha=0,
                                              rand=random.Random(SEED))
            else:
                walks = build_deepwalk_corpus_iter(G, num_paths=args.number_walks,
                                                   path_length=args.walk_length, alpha=0,
                                                   rand=random.Random(SEED),
                                                   chunk=epoch % chunks,
                                                   nodes=nodes_in_G)
            for walk in walks:
                if args.context == 1:
                    # Construct the pairs for predicting context node
                    # for each node, treated as center word
                    curr_pair = (int(walk[center_node_pos]), [])
                    for center_node_pos in range(len(walk)):
                        # for each window position
                        for w in range(-args.window_size, args.window_size + 1):
                            context_node_pos = center_node_pos + w
                            # make soure not jump out sentence
                            if context_node_pos < 0 or context_node_pos >= len(walk) or center_node_pos == context_node_pos:
                                continue
                            context_node_idx = walk[context_node_pos]
                            curr_pair[1].append(int(context_node_idx))
                else:
                    # first item in the walk is the starting node
                    curr_pair = (int(walk[0]), [int(context_node_idx) for context_node_idx in walk[1:]])

                if args.ns == 1:
                    neg_nodes = []
                    pos_nodes = set(walk)
                    while len(neg_nodes) < args.walk_length - 1:
                        rand_node = random.randint(0, n_nodes - 1)
                        if rand_node not in pos_nodes:
                            neg_nodes.append(rand_node)
                    neg_nodes = torch.from_numpy(np.array(neg_nodes)).long()

                # Do actual prediction
                src_node = torch.from_numpy(np.array([curr_pair[0]])).long()
                tgt_nodes = torch.from_numpy(np.array(curr_pair[1])).long()
                optimizer_dw.zero_grad()
                log_pos = sg(src_node, tgt_nodes, neg_sample=False)
                if args.ns == 1:
                    loss_neg = sg(src_node, neg_nodes, neg_sample=True)
                    loss_dw = log_pos + loss_neg
                else:
                    loss_dw = log_pos
                loss_dw.backward(retain_graph=True)
                cur_dw_loss = loss_dw.item()
                optimizer_dw.step()

        loss = loss_function(preds=model.dc(z), labels=adj_label,
                             mu=mu, logvar=logvar, n_nodes=n_nodes,
                             norm=norm, pos_weight=pos_weight)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)

        if args.dw == 1:
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, train_loss_dw={:.5f}, val_ap={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, cur_dw_loss,
                ap_curr, time.time() - t))
        else:
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, val_ap={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss,
                ap_curr, time.time() - t))

        if (epoch + 1) % 10 == 0:
            tqdm.write("Evaluating intermediate results...")
            kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
            predict_labels = kmeans.predict(hidden_emb)
            cm = clustering_metrics(true_labels, predict_labels)
            cm.evaluationClusterModelFromLabel(tqdm)
            roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
            tqdm.write('ROC: {}, AP: {}'.format(roc_score, ap_score))
            np.save('logs/emb_epoch_{}.npy'.format(epoch + 1), hidden_emb)

    tqdm.write("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)
    tqdm.write('Test ROC score: ' + str(roc_score))
    tqdm.write('Test AP score: ' + str(ap_score))
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit(hidden_emb)
    predict_labels = kmeans.predict(hidden_emb)
    cm = clustering_metrics(true_labels, predict_labels)
    cm.evaluationClusterModelFromLabel(tqdm)

    if args.plot == 1:
        cm.plotClusters(tqdm, hidden_emb, true_labels)


if __name__ == '__main__':
    gae_for(args)