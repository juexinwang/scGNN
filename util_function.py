import sys
import os
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.io
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from benchmark_util import *
from igraph import *
dir_path = os.path.dirname(os.path.realpath(__file__))


def checkargs(args):
    '''
    check whether paramters meets requirements
    '''
    # TODO
    return


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# Original version of load_data


def load_data_ori(datasetName, discreteTag):
    # load the data: x, tx, allx, graph
    if discreteTag:
        names = ['xD', 'txD', 'allxD', 'graph']
    else:
        names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open(dir_path+"/data/sc/{}/ind.{}.{}".format(datasetName, datasetName, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        dir_path+"/data/sc/{}/ind.{}.test.index".format(datasetName, datasetName))
    test_idx_range = np.sort(test_idx_reorder)

    if datasetName == 'citeseer':
        # Fix citeseer datasetName (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def load_data(datasetName, discreteTag):
    # load the data: x, tx, allx, graph
    if discreteTag:
        names = ['xD', 'txD', 'allxD']
    else:
        names = ['x', 'tx', 'allx']
    objects = []
    for i in range(len(names)):
        with open(dir_path+"/data/sc/{}/ind.{}.{}".format(datasetName, datasetName, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx = tuple(objects)
    test_idx_reorder = parse_index_file(
        dir_path+"/data/sc/{}/ind.{}.test.index".format(datasetName, datasetName))
    test_idx_range = np.sort(test_idx_reorder)

    if datasetName == 'citeseer':
        # Fix citeseer datasetName (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    return features

# TODO: transform does not work here, leave it, will work on it in next version


class logtransform(object):
    '''
    log transform of the object
    '''

    def __init__(self, sample):
        self.sample = sample

    def __call__(self, sample):
        return torch.log(sample)

# Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class scDatasetInter(Dataset):
    def __init__(self, features, transform=None):
        """
        Internal scData
        Args:
            construct dataset from features
        """
        self.features = features
        # Now lines are cells, and cols are genes
        # self.features = self.features.transpose()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class scBenchDataset(Dataset):
    def __init__(self, datasetName=None, discreteTag=False, transform=None):
        """
        For benchmark usage
        Args:
            datasetName (String): TGFb, etc.
            transform (callable, optional):
        """
        self.features = load_data(datasetName, discreteTag)
        # Now lines are cells, and cols are genes
        # self.features = self.features.transpose()
        # save nonzero
        self.nz_i, self.nz_j = self.features.nonzero()
        self.transform = transform
        # check whether log or not
        self.discreteTag = discreteTag

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        sample = torch.from_numpy(sample.toarray())

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        if not self.discreteTag:
            sample = torch.log(sample+1)

        return sample, idx


class scDatasetDropout(Dataset):
    def __init__(self, datasetName=None, discreteTag=False, ratio=0.1, seed=1, transform=None):
        """
        Args:
            datasetName (String): TGFb, etc.
            transform (callable, optional):
        """
        self.featuresOriginal = load_data(datasetName, discreteTag)
        self.ratio = ratio
        # Random seed
        # np.random.uniform(1, 2)
        self.features, self.i, self.j, self.ix = impute_dropout(
            self.featuresOriginal, seed=seed, rate=self.ratio)
        # Now lines are cells, and cols are genes
        # self.features = self.features.transpose()
        self.transform = transform
        # check whether log or not
        self.discreteTag = discreteTag

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        sample = torch.from_numpy(sample.toarray())

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        if not self.discreteTag:
            sample = torch.log(sample+1)

        return sample, idx


class scDataset(Dataset):
    def __init__(self, data=None, transform=None):
        """
        Args:
            data : sparse matrix.
            transform (callable, optional):
        """
        # Now lines are cells, and cols are genes
        self.features = data.transpose()

        # save nonzero
        # self.nz_i,self.nz_j = self.features.nonzero()
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx, :]
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        # transform after get the data
        if self.transform:
            sample = self.transform(sample)

        return sample, idx

# Original


def loss_function(recon_x, x, mu, logvar):
    '''
    Original: Classical loss function
    Reconstruction + KL divergence losses summed over all elements and batch
    '''
    # Original
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Graph


def loss_function_graph(recon_x, x, mu, logvar, graphregu=None, gammaPara=1.0, regulationMatrix=None, regularizer_type='noregu', reguPara=0.001, modelusage='AE', reduction='sum'):
    '''
    Regularized by the graph information
    Reconstruction + KL divergence losses summed over all elements and batch
    '''
    # Original
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Graph
    target = x
    if regularizer_type == 'Graph' or regularizer_type == 'LTMG' or regularizer_type == 'LTMG01':
        target.requires_grad = True
    # Euclidean
    # BCE = gammaPara * vallina_mse_loss_function(recon_x, target, reduction='sum')
    BCE = gammaPara * \
        vallina_mse_loss_function(recon_x, target, reduction=reduction)
    if regularizer_type == 'noregu':
        loss = BCE
    elif regularizer_type == 'LTMG':
        loss = BCE + reguPara * \
            regulation_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'LTMG01':
        loss = BCE + reguPara * \
            regulation01_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'Graph':
        loss = BCE + reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=graphregu, reduction=reduction)
    elif regularizer_type == 'GraphR':
        loss = BCE + reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=1-graphregu, reduction=reduction)
    elif regularizer_type == 'LTMG-Graph':
        loss = BCE + reguPara * regulation_mse_loss_function(recon_x, target, regulationMatrix, reduction=reduction) + \
            reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=graphregu, reduction=reduction)
    elif regularizer_type == 'LTMG-GraphR':
        loss = BCE + reguPara * regulation_mse_loss_function(recon_x, target, regulationMatrix, reduction=reduction) + \
            reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=1-graphregu, reduction=reduction)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if modelusage == 'VAE':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss + KLD

    return loss

# Graph


def loss_function_graph_celltype(recon_x, x, mu, logvar, graphregu=None, celltyperegu=None, gammaPara=1.0, regulationMatrix=None, regularizer_type='noregu', reguPara=0.001, reguParaCelltype=0.001, modelusage='AE', reduction='sum'):
    '''
    Regularized by the graph information
    Reconstruction + KL divergence losses summed over all elements and batch
    '''
    # Original
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Graph
    target = x
    if regularizer_type == 'Graph' or regularizer_type == 'LTMG' or regularizer_type == 'LTMG01' or regularizer_type == 'Celltype':
        target.requires_grad = True
    # Euclidean
    # BCE = gammaPara * vallina_mse_loss_function(recon_x, target, reduction='sum')
    BCE = gammaPara * \
        vallina_mse_loss_function(recon_x, target, reduction=reduction)
    if regularizer_type == 'noregu':
        loss = BCE
    elif regularizer_type == 'LTMG':
        loss = BCE + reguPara * \
            regulation_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'LTMG01':
        loss = BCE + reguPara * \
            regulation01_mse_loss_function(
                recon_x, target, regulationMatrix, reduction=reduction)
    elif regularizer_type == 'Graph':
        loss = BCE + reguPara * \
            graph_mse_loss_function(
                recon_x, target, graphregu=graphregu, reduction=reduction)
    elif regularizer_type == 'Celltype':
        loss = BCE + reguPara * graph_mse_loss_function(recon_x, target, graphregu=graphregu, reduction=reduction) + \
            reguParaCelltype * \
            graph_mse_loss_function(
                recon_x, target, graphregu=celltyperegu, reduction=reduction)
    elif regularizer_type == 'CelltypeR':
        loss = BCE + (1-gammaPara) * regulation01_mse_loss_function(recon_x, target, regulationMatrix, reduction=reduction) + reguPara * graph_mse_loss_function(recon_x,
                                                                                                                                                                 target, graphregu=graphregu, reduction=reduction) + reguParaCelltype * graph_mse_loss_function(recon_x, target, graphregu=celltyperegu, reduction=reduction)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if modelusage == 'VAE':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss + KLD

    return loss

# vallina mse
def vallina_mse_loss_function(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""vallina_mse_loss_function(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Original: Measures the element-wise mean squared error.

    See :revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    # Original, for not require grads, using c++ version
    # However, it has bugs there, different number of cpu cause different results because of MKL parallel library
    # Not known yet whether GPU has same problem.
    # Solution 1: set same number of cpu when running, it works for reproduce everything but not applicable for other users
    # https://pytorch.org/docs/stable/torch.html#torch.set_num_threads
    # https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
    # Solution 2: not use C++ codes, as we did here.
    # https://github.com/pytorch/pytorch/issues/8710

    if target.requires_grad:
        ret = (input - target) ** 2
        # 0.001 to reduce float loss
        # ret = (0.001*input - 0.001*target) ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(
            input, target)
        ret = torch._C._nn.mse_loss(
            expanded_input, expanded_target, get_enum(reduction))

    # ret = (input - target) ** 2
    # # 0.001 to reduce float loss
    # # ret = (0.001*input - 0.001*target) ** 2
    # if reduction != 'none':
    #     ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

# Regulation mse as the regularizor
# Now LTMG is set as the input


def regulation_mse_loss_function(input, target, regulationMatrix, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, str, Optional[bool], Optional[bool], str) -> Tensor
    r"""regulation_mse_loss_function(input, target, regulationMatrix, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error for regulation input, now only support LTMG.

    See :revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    ret = (input - target) ** 2
    # ret = (0.001*input - 0.001*target) ** 2
    ret = torch.mul(ret, regulationMatrix)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

# Regulation mse as the regularizor
# Now LTMG is set as the input


def regulation01_mse_loss_function(input, target, regulationMatrix, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, str, Optional[bool], Optional[bool], str) -> Tensor
    r"""regulation_mse_loss_function(input, target, regulationMatrix, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error for regulation input, now only support LTMG.

    See :revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    ret = (input - target) ** 2
    # ret = (0.001*input - 0.001*target) ** 2
    regulationMatrix[regulationMatrix > 0] = 1
    ret = torch.mul(ret, regulationMatrix)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def graph_mse_loss_function(input, target, graphregu, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""graph_mse_loss_function(input, target, adj, regularizer_type, size_average=None, reduce=None, reduction='mean') -> Tensor
    Measures the element-wise mean squared error in graph regularizor.
    See:revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
              "This will likely lead to incorrect results due to broadcasting. "
              "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    # Now it use regulariz type to distinguish, it can be imporved later
    ret = (input - target) ** 2
    # ret = (0.001*input - 0.001*target) ** 2
    # if graphregu != None:
    # print(graphregu.type())
    # print(ret.type())
    ret = torch.matmul(graphregu, ret)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def legacy_get_enum(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> int
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))

# We use these functions in torch/legacy as well, in which case we'll silence the warning


def legacy_get_string(size_average, reduce, emit_warning=True):
    # type: (Optional[bool], Optional[bool], bool) -> str
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = 'mean'
    elif reduce:
        ret = 'sum'
    else:
        ret = 'none'
    if emit_warning:
        print(warning.format(ret))
    return ret


def get_enum(reduction):
    # type: (str) -> int
    if reduction == 'none':
        ret = 0
    elif reduction == 'mean':
        ret = 1
    elif reduction == 'elementwise_mean':
        print("reduction='elementwise_mean' is deprecated, please use reduction='mean' instead.")
        ret = 1
    elif reduction == 'sum':
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(
            "{} is not a valid value for reduction".format(reduction))
    return ret


def save_sparse_matrix(filename, x):
    x_coo = x.tocoo()
    row = x_coo.row
    col = x_coo.col
    data = x_coo.data
    shape = x_coo.shape
    np.savez(filename, row=row, col=col, data=data, shape=shape)


def load_sparse_matrix(filename):
    y = np.load(filename)
    z = scipy.sparse.coo_matrix(
        (y['data'], (y['row'], y['col'])), shape=y['shape'])
    return z


def trimClustering(listResult, minMemberinCluster=5, maxClusterNumber=30):
    '''
    If the clustering numbers larger than certain number, use this function to trim. May have better solution
    '''
    numDict = {}
    for item in listResult:
        if not item in numDict:
            numDict[item] = 0
        else:
            numDict[item] = numDict[item]+1

    size = len(set(listResult))
    changeDict = {}
    for item in range(size):
        if numDict[item] < minMemberinCluster or item >= maxClusterNumber:
            changeDict[item] = ''

    count = 0
    for item in listResult:
        if item in changeDict:
            listResult[count] = maxClusterNumber
        count += 1

    return listResult


def readLTMG(LTMGDir, ltmgfile):
    '''
    Read LTMG matrix as the regularizor. sparseMode for huge datasets sparse coding, now only use sparseMode
    '''
    # sparse mode
    # if sparseMode:
    df = pd.read_csv(LTMGDir+ltmgfile, header=None,
                     skiprows=1, delim_whitespace=True)
    for row in df.itertuples():
        # For the first row, it contains the number of genes and cells. Init the whole matrix
        if row[0] == 0:
            matrix = np.zeros((row[2], row[1]))
        else:
            matrix[row[2]-1][row[1]-1] = row[3]
    # nonsparse mode: read in csv format, very very slow when the input file is huge, not using
    # else:
    #     matrix = pd.read_csv(LTMGDir+ltmgfile,header=None, index_col=None, delimiter='\t', engine='c')
    #     matrix = matrix.to_numpy()
    #     matrix = matrix.transpose()
    #     matrix = matrix[1:,1:]
    #     matrix = matrix.astype(int)
    return matrix


def readLTMGnonsparse(LTMGDir, ltmgfile):
    '''
    Read LTMG matrix as the regularizor. nonsparseMode
    '''
    # nonsparse mode: read in csv format, very very slow when the input file is huge, not using
    matrix = pd.read_csv(LTMGDir+ltmgfile, header=None,
                         index_col=None, delimiter='\t', engine='c')
    matrix = matrix.to_numpy()
    matrix = matrix.transpose()
    matrix = matrix[1:, 1:]
    matrix = matrix.astype(int)
    return matrix


def loadscExpression(csvFilename, sparseMode=True):
    '''
    Load sc Expression: rows are genes, cols are cells, first col is the gene name, first row is the cell name.
    sparseMode for loading huge datasets in sparse coding
    '''
    if sparseMode:
        print('Load expression matrix in sparseMode')
        genelist = []
        celllist = []
        with open(csvFilename.replace('.csv', '_sparse.npy'), 'rb') as f:
            objects = pkl.load(f, encoding='latin1')
        matrix = objects.tolil()

        with open(csvFilename.replace('.csv', '_gene.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                genelist.append(line)

        with open(csvFilename.replace('.csv', '_cell.txt')) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                celllist.append(line)

    else:
        print('Load expression in csv format')
        matrix = pd.read_csv(csvFilename, index_col=0)
        genelist = matrix.index.tolist()
        celllist = matrix.columns.values.tolist()
        matrix = matrix.to_numpy()
        matrix = matrix.astype(float)

    return matrix, genelist, celllist


def generateCelltypeRegu(listResult):
    celltypesample = np.zeros((len(listResult), len(listResult)))
    tdict = {}
    count = 0
    for item in listResult:
        if item in tdict:
            tlist = tdict[item]
        else:
            tlist = []
        tlist.append(count)
        tdict[item] = tlist
        count += 1

    for key in sorted(tdict):
        tlist = tdict[key]
        for item1 in tlist:
            for item2 in tlist:
                celltypesample[item1, item2] = 1.0

    return celltypesample


def generateLouvainCluster(edgeList):
    """
    Louvain Clustering using igraph
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {}
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = []
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size
