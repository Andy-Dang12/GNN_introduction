import torch, dgl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    def init(self, in_feat:int, out_feat:int, p:float=0.05):
        super(SAGEConv, self).init()
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
    def init(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).init()
        self.emb = nn.Embedding(67, in_feats)
        self.conv1 = SAGEConv(in_feats + 2, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)
        
    def build_graph(self, inputs):
        nodes_labels = [inp[-1] for inp in inputs]
        nodes_label = np.eye(2)[nodes_labels]
        nodes_feat = torch.stack([torch.concat((torch.Tensor(inp[0]), torch.sum(embed(torch.LongTensor(inp[1])), dim = 0))) for inp in inputs], dim = 0)
        g = dgl.graph((range(0, len(inputs)-1), range(1, len(inputs))), num_nodes=len(inputs))
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)
        g.ndata['feat' ] = nodes_feat
        g.ndata['label'] = torch.tensor(nodes_label)
        return g, torch.Tensor(nodes_labels)
    
    def forward(self, inputs):
        g, label = self.build_graph(inputs)
        in_feat = g.ndata['feat'].float()
        lbl = g.ndata['label']
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h, lbl, label