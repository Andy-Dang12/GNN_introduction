import torch, dgl
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import CoraGraphDataset

dataset = CoraGraphDataset()
# print('Number of categories:', dataset.num_classes)

#NOTE cora have only one graph
g = dataset[0]      # def __getitem__(self, idx) assert idx == 0, "This dataset has only one graph"


def show_graph_data():
    #NOTE show node/edge feature
    print('Node features\n')
    feat = g.ndata['feat'][0]
    print(feat.shape)
    # for k, v in g.ndata.items():
    #     print(k)
    # print('Edge features')
    # print(g.edata)
show_graph_data()

#NOTE defining a GCN
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)

