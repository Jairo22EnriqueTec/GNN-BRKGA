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

parser.add_argument("-pi", "--PATH", help = "", type = str)
parser.add_argument("-pv", "--PATH_VAL", help = "", type = str)
parser.add_argument("-ps", "--PATH_SAVE", help = "", type=str)
parser.add_argument("-MDH", "--MDH", help = "", type = int)
parser.add_argument("-s", "--seed", help = "", type = int)
parser.add_argument("-e", "--epochs", help = "", type = int)

args = parser.parse_args()

PATH_TO_TRAIN = args.PATH
PATH_TO_VALIDATION = args.PATH_VAL
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
    num_features = 4 # Change if needed
    
num_classes = 2

threshold = 0.5

optimizer_name = "Adam"
lr = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

layers = ["GCN", "GAT","GraphConv", "SAGE"]
#layers = ["SAGE"]

Models = [GNNModel(c_in = num_features, c_hidden = 100, c_out = num_classes, num_layers = 2, layer_name = layer_name, dp_rate=0.1) for 
         layer_name in layers]


# In[17]:

# Training instances
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

# validation instances
if PATH_TO_VALIDATION != "":
    Instances = [graph for graph in os.listdir(PATH_TO_VALIDATION + 'txt')]

    graphs_val = []
    for er in Instances:
        graph = igraph.Graph.Read_Edgelist(PATH_TO_VALIDATION+"txt/"+er, directed = False)
        graphs_val.append(graph.to_networkx())    


    OptInstances = [graph for graph in os.listdir(PATH_TO_VALIDATION+'optimal')]
    Solutions_val = []
    for er in OptInstances:
        opt = []
        with open(PATH_TO_VALIDATION+'optimal/'+er) as f:
            for line in f.readlines():
                opt.append(int(line.replace("\n", "")))
        Solutions_val.append(opt)
    
   

if not MDH:
    print("\nCargando Features...\n")
    graphFeatures = [feat for feat in os.listdir(PATH_TO_TRAIN+'feats')]
    Features = []
    for er in graphFeatures:
        temp = []
        try:
            with open(PATH_TO_TRAIN+'feats/'+er) as f:
                c = 0

                for line in f.readlines()[1:]:
                    c+=1
                    feats = np.array(line.split(","), dtype = float)
                    temp.append(feats)
            Features.append(np.array(temp))
        except:
            print(er)
            print(line)
            print(c)
        
    if PATH_TO_VALIDATION != "":
        graphFeatures_val = [feat for feat in os.listdir(PATH_TO_VALIDATION+'feats')]
        Features_val = []
        for er in graphFeatures_val:
            temp = []
            with open(PATH_TO_VALIDATION+'feats/'+er) as f:

                for line in f.readlines()[1:]:
                    feats = np.array(line.split(","), dtype = float)
                    temp.append(feats)
            Features_val.append(np.array(temp))
# In[19]:


# In[20]:

# Se escalan en la clase
if PATH_TO_VALIDATION != "":
    Graphs_Val = Convert2DataSet(graphs_val, Solutions_val, feats = Features_val)
else:
    Graphs_Val = ""
    
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
    
    ES = EarlyStopping(12, PATH_SAVE_TRAINS, layers[i], SEED, threshold, epochs)
    
    for epoch in range(1, epochs):
        
        for data in Graphs_Train:
            train(Models[i], optimizer, data)
        
        if epoch%10 == 0:
            #torch.save(Models[i].state_dict(), f=f"{PATH_SAVE_TRAINS_CHECKPOINTS}Checkpoint-model-{layers[i]}-epoch-{epoch}.pt")
            print(f"Epoch {epoch} saved for {layers[i]}.\n")
        
            Acc = []

            for data in Graphs_Train:
                Acc.append(test(data, Models[i]))
            print(f"Mean Acc: {np.mean(Acc)}")
            print()
        
        
        if ES.check(Models[i], Graphs_Val, Graphs_Train):
            break
    
    print(f"{layers[i]} saved in {PATH_SAVE_TRAINS}\n")
    #ES.plot_history(layers[i])


# In[ ]:




