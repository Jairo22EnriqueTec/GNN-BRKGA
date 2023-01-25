import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import Linear

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, name_layer = "SAGE"):
        super().__init__()
        self.name = name_layer
        layer = None
        hidden_feats = 8
        
        if name_layer == "SAGE":
            layer = geom_nn.SAGEConv
            
            self.conv1 = layer(num_node_features, num_classes)
            #self.conv3 = layer(hidden_feats, num_classes)
            
        elif name_layer == "GAT":
            layer = geom_nn.GATConv
            
            self.conv1 = layer(num_node_features, num_classes)
            #self.conv3 = layer(hidden_feats, num_classes)
            
        elif name_layer == "GCN":
            layer = geom_nn.GCNConv
            
            self.conv1 = layer(num_node_features, num_classes)
            #self.conv1 = layer(num_node_features, hidden_feats)
            #self.conv3 = Linear(hidden_feats, num_classes)
            
        elif name_layer == "GraphConv":
            layer = geom_nn.GraphConv
            
            self.conv1 = layer(num_node_features, num_classes)
            
        elif name_layer == "SGConv":
            layer = geom_nn.SGConv
            
            self.conv1 = layer(num_node_features, num_classes)
            #self.conv3 = Linear(hidden_feats, num_classes)
        
        
        else:
            print("Nanais")
    
        
        #self.conv2 = layer(hidden_feats, hidden_feats)
        #self.conv3 = Linear(hidden_feats, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)

        #return F.log_softmax(self.conv3(x), dim=1)
        #if self.name in ['GraphConv', 'GCN', 'SAGE']:
        return F.log_softmax(x, dim=1)
        #else:
        #    return F.log_softmax(self.conv3(x), dim=1)
        