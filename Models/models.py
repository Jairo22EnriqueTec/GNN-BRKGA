import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import Linear

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, name_layer = "SAGE", num_layers = 1):
        super().__init__()
        self.name = name_layer
        self.num_layers = num_layers
        layer = None
        hidden_feats = 8
        
        if name_layer == "SAGE":
            layer = geom_nn.SAGEConv
            
        elif name_layer == "GAT":
            layer = geom_nn.GATConv
            
        elif name_layer == "GCN":
            layer = geom_nn.GCNConv
            
        elif name_layer == "GraphConv":
            layer = geom_nn.GraphConv
            
        elif name_layer == "SGConv":
            layer = geom_nn.SGConv
            
        else:
            print("Nanais")
        
        if self.num_layers == 1:
            self.conv1 = layer(num_node_features, num_classes)
        elif self.num_layers == 2:
            self.conv1 = layer(num_node_features, hidden_feats)
            self.conv2 = layer(hidden_feats, num_classes)
        elif self.num_layers == 3:
            self.conv1 = layer(num_node_features, hidden_feats)
            self.conv2 = layer(hidden_feats, hidden_feats)
            self.conv3 = layer(hidden_feats, num_classes)
        
        
        #self.conv2 = layer(hidden_feats, hidden_feats)
        #self.conv3 = Linear(hidden_feats, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        if self.num_layers == 1:
            pass
        elif self.num_layers == 2:
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        elif self.num_layers == 3:
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
        
        return F.log_softmax(x, dim=1)
            