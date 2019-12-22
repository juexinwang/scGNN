import sys
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

#Original version of load_data
def load_data_ori(datasetName, discreteTag):
    # load the data: x, tx, allx, graph
    if discreteTag:
        names = ['xD', 'txD', 'allxD', 'graph']
    else:
        names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/sc/{}/ind.{}.{}".format(datasetName, datasetName, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/sc/{}/ind.{}.test.index".format(datasetName, datasetName))
    test_idx_range = np.sort(test_idx_reorder)

    if datasetName == 'citeseer':
        # Fix citeseer datasetName (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
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
        with open("data/sc/{}/ind.{}.{}".format(datasetName, datasetName, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx = tuple(objects)
    test_idx_reorder = parse_index_file("data/sc/{}/ind.{}.test.index".format(datasetName, datasetName))
    test_idx_range = np.sort(test_idx_reorder)

    if datasetName == 'citeseer':
        # Fix citeseer datasetName (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    return features

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

        sample = self.features[idx,:]

        if self.transform:
            sample = self.transform(sample)

        sample = torch.from_numpy(sample.toarray())
        return sample

class scDataset(Dataset):
    def __init__(self, datasetName=None, discreteTag=False, transform=None):
        """
        Args:
            datasetName (String): TGFb, etc.
            transform (callable, optional):
        """
        self.features = load_data(datasetName,discreteTag)
        # Now lines are cells, and cols are genes
        # self.features = self.features.transpose()
        self.transform = transform        

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx,:]

        if self.transform:
            sample = self.transform(sample)

        sample = torch.from_numpy(sample.toarray())
        return sample

# Original
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    # Original 
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Reconstruction + KL divergence losses summed over all elements and batch
# graph
def loss_function_graph(recon_x, x, mu, logvar, adjsample, adjfeature, regulized_type, modelusage):
    # Original 
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Graph
    target = x
    # target.requires_grad = True
    # Euclidean
    BCE = graph_mse_loss_function(recon_x, target, adjsample, adjfeature, regulized_type='noregu', reduction='sum')
    # Entropy
    # BCE = graph_binary_cross_entropy(recon_x, target, adj, reduction='sum')
    # BCE = F.binary_cross_entropy(recon_x, target, reduction='sum')
    loss = BCE

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if modelusage == 'VAE':
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD    

    return loss

# change from pytorch
# Does not work now
# def graph_binary_cross_entropy(input, target, adj, weight=None, size_average=None,
#                          reduce=None, reduction='mean'):
#     # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
#     r"""Function that measures the Binary Cross Entropy
#     between the target and the output.

#     See :class:`~torch.nn.BCELoss` for details.

#     Args:
#         input: Tensor of arbitrary shape
#         target: Tensor of the same shape as input
#         weight (Tensor, optional): a manual rescaling weight
#                 if provided it's repeated to match input tensor shape
#         size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
#             the losses are averaged over each loss element in the batch. Note that for
#             some losses, there multiple elements per sample. If the field :attr:`size_average`
#             is set to ``False``, the losses are instead summed for each minibatch. Ignored
#             when reduce is ``False``. Default: ``True``
#         reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
#             losses are averaged or summed over observations for each minibatch depending
#             on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
#             batch element instead and ignores :attr:`size_average`. Default: ``True``
#         reduction (string, optional): Specifies the reduction to apply to the output:
#             ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
#             ``'mean'``: the sum of the output will be divided by the number of
#             elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
#             and :attr:`reduce` are in the process of being deprecated, and in the meantime,
#             specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

#     Examples::

#         >>> input = torch.randn((3, 2), requires_grad=True)
#         >>> target = torch.rand((3, 2), requires_grad=False)
#         >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
#         >>> loss.backward()
#     """
#     if size_average is not None or reduce is not None:
#         reduction_enum = legacy_get_enum(size_average, reduce)
#     else:
#         reduction_enum = get_enum(reduction)
#     if target.size() != input.size():
#         print("Using a target size ({}) that is different to the input size ({}) is deprecated. "
#                       "Please ensure they have the same size.".format(target.size(), input.size()),
#                       stacklevel=2)
#     if input.numel() != target.numel():
#         raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
#                          "!= input nelement ({})".format(target.numel(), input.numel()))

#     if weight is not None:
#         # new_size = _infer_size(target.size(), weight.size())
#         # weight = weight.expand(new_size)
#         print("Not implement yet from pytorch")

#     if args.regulized_type == 'Graph':
#         target.requires_grad = True
#         input = torch.matmul(input, adj)
#         target = torch.matmul(target, adj)

#     return torch._C._nn.binary_cross_entropy(
#         input, target, weight, reduction_enum)

# graphical mse
def graph_mse_loss_function(input, target, adjsample, adjfeature, regulized_type='noregu', size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""graph_mse_loss_function(input, target, adj, regulized_type, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :revised from pytorch class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        print("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if size_average is not None or reduce is not None:
        reduction = legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = (input - target) ** 2
        #key is here
        if regulized_type == 'Graph':
            if adjsample != None:
                ret = torch.matmul(adjsample, ret)
            if adjfeature != None:
                ret = torch.matmul(ret, adjfeature)
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, get_enum(reduction))
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
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret