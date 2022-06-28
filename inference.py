from typing import Tuple
import torch, dgl
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
import re
import numpy as np
import pandas as pd
import os.path as osp
from colorama import Fore
from glob import glob
from random import shuffle
import warnings
warnings.filterwarnings('ignore')


class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    
    def __init__(self, in_feat:int, out_feat:int):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(message_func = fn.copy_u('h', 'm'), 
                         reduce_func = fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
num_classes = 48
model = GraphSAGE(772, 128, num_classes)

def _get_n_nodes(nodes_label:pd.DataFrame) -> int:
    r"""
    tính và kiểm tra số thứ tự của node
    """
    n_nodes = nodes_label['Id'].to_list()
    for i, idx in enumerate(n_nodes):
        assert i == idx, 'i != idx'
    return len(n_nodes)


def build_graph(edgep:str):
    nodes_feat = np.load(re.sub('.edges.csv$', '.nfeat.npy', edgep))
    nodes_label = pd.read_csv(
        re.sub('.edges.csv$', '.idx.csv', edgep), encoding='utf-8')
    n_nodes = _get_n_nodes(nodes_label)
    
    nodes_label = nodes_label['label'].astype('category').cat.codes.to_list()
    edge = pd.read_csv(edgep, encoding='utf-8')
    
    g = dgl.graph((edge['src'], edge['dst']), num_nodes=n_nodes)
    g = dgl.to_bidirected(g)
    g = dgl.remove_self_loop(g)
    g.ndata['feat' ] = torch.from_numpy(nodes_feat )
    g.ndata['label'] = torch.tensor    (nodes_label)
    return g

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

@torch.no_grad()
def inference(model:nn.Module, g:dgl.DGLGraph) -> Tuple[int, float]:
    features = g.ndata['feat'].float()
    # labels = g.ndata['label']
    logits = model(g, features)
    pred = logits.argmax(dim=1)
    score = F.softmax(logits).max(dim=1)
    return pred, score


g = build_graph('dataset/DKKD_graph_test/0002.edges.csv')
pred, score = inference(model, g)

print(score)
