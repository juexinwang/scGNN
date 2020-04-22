import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, input_feat_dim, num_nodes):
        super(SkipGram, self).__init__()
        self.num_nodes = num_nodes
        """ word embeddings """
        self.word_embedding = torch.nn.Embedding(num_nodes, input_feat_dim)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.word_embedding.weight)
        """ context embeddings"""
        self.context_embedding = torch.nn.Embedding(num_nodes, input_feat_dim)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.context_embedding.weight)

    def get_input_layer(self, word_idx):
        x = torch.zeros(self.num_nodes).float()
        x[word_idx] = 1.0
        return x

    def forward(self, node, context_positions, neg_sample=False):

        embed_word = self.word_embedding(node)  # 1 * emb_size
        embed_context = self.context_embedding(context_positions)  # n * emb_size
        score = torch.matmul(embed_context, embed_word.transpose(dim0=1, dim1=0))  # score = n * 1

        # following is an example of something you can only do in a framework that allows
        # dynamic graph creation
        if neg_sample:
            score = -1 * score
        obj = -1 * torch.sum(F.logsigmoid(score))

        return obj
