import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax
# Guanhao's GRAT
class GRATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        # TODO: why not another linear layer before entering the net
        super(GRATLayer, self).__init__()
        # linear layer
        self.fc = nn.Linear(in_feats, out_feats, bias=True)  # bias=True in Guanhao's original code
        
        # attention layer
        self.attn_fc = nn.Linear(2 * in_feats, 1, bias=True)  # bias=True in Guanhao's original code

        # initialize parameters
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        h2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.attn_fc(h2)
        return {'e': torch.relu(a)}

    def message_func(self, edges):
        return {'h': edges.src['h'] * edges.data['alpha']}  # message divided by weight

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            # Equation (2)
            g.apply_edges(self.edge_attention)  # calculate e_{ij}

            # Calculate softmax on source code -> on the reversed graph
            rg = g.reverse(copy_ndata=False, copy_edata=True)
            g.edata['alpha'] = edge_softmax(rg, rg.edata['e'])
            
            # Convolution
            g.update_all(self.message_func, self.reduce_func)
            
            g.ndata['h'] = self.fc(g.ndata['h'])
            h = g.ndata['h']
            return h


class GRAT3_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, out_feats, *args):
        super(GRAT3_, self).__init__()
        self.grat1 = GRATLayer(in_feats, hidden_feats1)
        self.grat2 = GRATLayer(hidden_feats1, hidden_feats2)
        self.grat3 = GRATLayer(hidden_feats2, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))
        h = torch.relu(self.grat2(g, h))
        h = self.grat3(g, h)
        return h


class GRAT3(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, *args):
        super(GRAT3, self).__init__()
        self.grat = GRAT3_(in_feats, hidden_feats1, hidden_feats2, 1, *args)  # out_feats=1

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h