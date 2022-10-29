import networkx as nx
import numpy as np
import torch
from datetime import datetime

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

import pandas as pd
import torch_geometric.transforms as T

from create_dataset import CreateDataset

from models import GNNModel

import sys
sys.path.append("../FastCover/")
from utils import *
import warnings
warnings.filterwarnings('ignore')
import argparse



parser = argparse.ArgumentParser()

parser.add_argument("-s", "--Seed", help = "", type=int)
parser.add_argument("-p", "--Prob", help = "", type=float)
parser.add_argument("-n", "--Nodos", help = "", type=int)
parser.add_argument("-e", "--Epochs", help = "", type=int)
parser.add_argument("-th", "--Threshold", help = "", type=float)
parser.add_argument("-lr", "--LearningRate", help = "", type=float)
args = parser.parse_args()

#python TrainModels.py -th 0.1 -p 0.5 -n 10 -e 3 -lr 0.0005 -s 666

PATH_SAVE_TRAINS_CHECKPOINTS = 'runs/checkpoints/'
PATH_SAVE_TRAINS = 'runs/'

num_features = 1
num_classes  = 2

threshold = args.Threshold

optimizer_name = 'Adam'
lr = args.LearningRate
epochs = args.Epochs

SEED = args.Seed

n = args.Nodos

p = args.Prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layers = ["GCN", "GAT","GraphConv", "SAGE"]

Models = [GNNModel(c_in = 1, c_hidden = 100, c_out = 2, num_layers = 2, layer_name = layer_name, dp_rate=0.1) for 
         layer_name in layers]

def getGraph(n, p = 0.5):
    while True:
        G = nx.erdos_renyi_graph(n, p, directed = False)
        AllConected = True if len([i for i in nx.connected_components(G)]) == 1 else False
        if AllConected:
            break
    return G

def Convert2DataSet(Graphs):
    g = []
    for G in Graphs:
        # Cambiar aquí la forma de cómo calcular dichos valores
        OptimalSet, _ = MDH(G, threshold, print_= False)

        NumNodes = G.number_of_nodes()
        labels = np.zeros(NumNodes)
        labels[OptimalSet] = 1

        dataset = CreateDataset(G, labels)
        data = dataset[0]

        data =  data.to(device)
        g.append(data)
    return g


Train = [getGraph(n+i, p) for i in range(10)]

Graphs_Train = Convert2DataSet(Train)

def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()

        F.nll_loss(model(data.x, data.edge_index), data.y).backward()
        optimizer.step()
        return model, optimizer
      
    
@torch.no_grad()
def test(data, model):
  model.eval()
  logits = model(data.x, data.edge_index)
  pred = logits.max(1)[1]
  acc = pred.eq(data.y).sum().item() / data.num_nodes
  return acc



torch.manual_seed(SEED)
for i in range(len(Models)):
    print()
    print(f" ----- Model:{layers[i]} -----")
    optimizer = getattr(torch.optim, optimizer_name)(Models[i].parameters(), lr = lr)

    for epoch in range(1, epochs+1):
        
        for data in Graphs_Train:
            train(Models[i], optimizer, data)
        
        if epoch%5 == 0:
            torch.save(Models[i].state_dict(), f=f"{PATH_SAVE_TRAINS_CHECKPOINTS}Checkpoint-model-{layers[i]}-epoch-{epoch}.pt")
            print(f"Epoch {epoch} saved for {layers[i]}.\n")
        
            Acc = []

            for data in Graphs_Train:
                Acc.append(test(data, Models[i]))
            print(f"Mean Acc: {np.mean(Acc)}")
            print()
        
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    torch.save(Models[i].state_dict(), f=f"{PATH_SAVE_TRAINS}{layers[i]}_seed_{SEED}_thr_{int(threshold*10)}_date_{dt_string}.pt")
    
    print(f"{layers[i]} saved in {PATH_SAVE_TRAINS}\n")