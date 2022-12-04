#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import numpy as np
import torch
from datetime import datetime
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from icecream import ic

ic.configureOutput("debug | -> ")

import pandas as pd
import torch_geometric.transforms as T


from models import GNNModel

import sys
sys.path.append("../FastCover/")
from utils import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--PATH", help = "", type = str)
parser.add_argument("-ps", "--PATH_SAVE", help = "", type=str)
parser.add_argument("-MDH", "--MDH", help = "", type = bool)
parser.add_argument("-s", "--seed", help = "", type = int)
parser.add_argument("-e", "--epochs", help = "", type = int)

args = parser.parse_args()

PATH_TO_TRAIN = args.PATH
PATH_SAVE_TRAINS = args.PATH_SAVE
MDH = bool(args.MDH)
epochs = args.epochs
SEED = args.seed

#v.g. python TrainModels.py -p "../BRKGA/instances/Erdos/train/" -ps "runs/Erdos_MDH/" -MDH 1 -s 13 -e 21

#PATH_SAVE_TRAINS = 'runs/Erdos_MDH/'
#PATH_TO_TRAIN = "../BRKGA/instances/Erdos/train/"


Features = None
if MDH:
    num_features = 1
else:
    num_features = 5 # Change if needed
    
num_classes  = 2

threshold = 0.5

optimizer_name = "Adam"
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layers = ["GCN", "GAT","GraphConv", "SAGE"]
#layers = ["SAGE"]

Models = [GNNModel(c_in = num_features, c_hidden = 100, c_out = 2, num_layers = 2, layer_name = layer_name, dp_rate=0.1) for 
         layer_name in layers]


# In[17]:


Instances = [graph for graph in os.listdir(PATH_TO_TRAIN + 'txt')]

graphs = []
for er in Instances:
    graph = igraph.Graph.Read_Edgelist(PATH_TO_TRAIN+"txt/"+er, directed = False)
    graphs.append(graph.to_networkx())    

OptInstances = [graph for graph in os.listdir(PATH_TO_TRAIN+'optimal')]
Solutions = []
for er in OptInstances:
    opt = []
    with open(PATH_TO_TRAIN+'optimal/'+er) as f:
        for line in f.readlines():
            opt.append(int(line.replace("\n", "")))
    Solutions.append(opt)


# In[18]:


if not MDH:
    graphFeatures = [feat for feat in os.listdir(PATH_TO_TRAIN+'feats')]
    Features = []
    for er in graphFeatures:
        temp = []
        with open(PATH_TO_TRAIN+'feats/'+er) as f:
            for line in f.readlines()[1:]:
                feats = np.array(line.split(","), dtype = float)
                temp.append(feats)
        Features.append(np.array(temp))


# In[19]:


# In[20]:

# Se escalan en la clase
Graphs_Train = Convert2DataSet(graphs, Solutions, feats = Features)
num_features = Graphs_Train[0].num_features
num_classes = Graphs_Train[0].num_classes


# ## Train

# In[21]:


def train(model, optimizer, data):
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


# In[22]:


torch.manual_seed(SEED)
for i in range(len(Models)):
    print()
    print(f" ----- Model:{layers[i]} -----")
    optimizer = getattr(torch.optim, optimizer_name)(Models[i].parameters(), lr = lr)

    for epoch in range(1, epochs):
        
        for data in Graphs_Train:
            train(Models[i], optimizer, data)
        
        if epoch%5 == 0:
            #torch.save(Models[i].state_dict(), f=f"{PATH_SAVE_TRAINS_CHECKPOINTS}Checkpoint-model-{layers[i]}-epoch-{epoch}.pt")
            print(f"Epoch {epoch} saved for {layers[i]}.\n")
        
            Acc = []

            for data in Graphs_Train:
                Acc.append(test(data, Models[i]))
            print(f"Mean Acc: {np.mean(Acc)}")
            print()
        
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    torch.save(Models[i].state_dict(), f=f"{PATH_SAVE_TRAINS}{layers[i]}_seed_{SEED}_thr_{int(threshold*10)}_date_{dt_string}.pt")
    
    print(f"{layers[i]} saved in {PATH_SAVE_TRAINS}\n")


# In[ ]:




