import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    
    def __init__(self, in_feat:int, out_feat:int, p:float=0.05):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)
        self.dropout = nn.Dropout(p=p)
        
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
                        #  reduce_func = fn.max('m', 'h_N'))
                        #  reduce_func = fn.sum('m', 'h_N'))
                        #  reduce_func = fn.sum('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            
            return self.linear(self.dropout(h_total))


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats_1, h_feats_2, p, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats_1, p)
        self.conv2 = SAGEConv(h_feats_1, h_feats_2, p)
        self.conv3 = SAGEConv(h_feats_2, num_classes, p)
        

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h