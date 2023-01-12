#!/usr/bin/env python
# coding: utf-8

# In[10]:


import networkx as nx
import numpy as np
import torch
from datetime import datetime
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn as geom_nn

from geneticalgorithm import geneticalgorithm as ga
import sys


import pandas as pd
import torch_geometric.transforms as T

from models import GNN

sys.path.append("../FastCover/")

from utils import *


# In[12]:


import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-pi", "--PATH", help = "", type = str)
parser.add_argument("-ps", "--PATH_SAVE", help = "", type=str)
parser.add_argument("-s", "--seed", help = "", type = int)
parser.add_argument("-i", "--iterations", help = "", type = int)
parser.add_argument("-pop", "--popsize", help = "", type = int)
parser.add_argument("-elit", "--elitratio", help = "", type = float)

args = parser.parse_args()

PATH_TO_TRAIN = args.PATH
PATH_SAVE_TRAINS = args.PATH_SAVE
SEED = args.seed

max_iterations = args.iterations
pop_size = args.popsize
elitratio = args.elitratio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[38]:

# python GA_Training.py -pi "../BRKGA/instances/scalefree/train/" -ps "./runs/scalefree/GA/" -s 13 -i 100 -pop 20 -elit 0.01


threshold = 0.5

dt_string = datetime.now().strftime("%m-%d_%H-%M")


# In[39]:


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
        temp = np.array(temp)
        #temp = np.delete(temp, 2, 1)
        Features.append(temp)
    except:
        print(er)
        print(line)
        print(c)
    
Graphs_Train = Convert2DataSet(graphs, Solutions, feats = Features)

num_features = Graphs_Train[0].num_features
num_classes = Graphs_Train[0].num_classes


# In[40]:


layers = ["GCN", "GAT","GraphConv", "SAGE", "SGConv"]

torch.manual_seed(SEED)
Models = [GNN(num_features, num_classes, name_layer = layer_name) for 
         layer_name in layers]


# In[42]:


def getDimParams(model):
    # Returns the number of parameters needed
    sum_ = 0
    for name, param in model.named_parameters():
        m = 1
        #print(name, param.shape)
        #print(np.max(param.detach().numpy()), np.min(param.detach().numpy()))
        for n in param.shape:
            m*=n
        sum_ += m
    return sum_ 

def getStateDict(model, params):
    # reensamble the original state dict with new values
    sd = model.state_dict()
    sum_ = 0
    from_ = 0

    for name, param in model.named_parameters():
        m = 1
        for n in param.shape:
            m*=n
        #print(m)
        #print(vals[from_:from_+m].reshape(param.shape).shape)
        sd[name] = torch.tensor(params[from_:from_+m].reshape(param.shape))
        from_ += m
        #print(from_)
    return sd

def Func(X, MDH = False):
    # Objective function
    if not MDH:
        sd = getStateDict(Models[i], X)
        Models[i].load_state_dict(sd)
    
    value = 0.0
    
    for ig, data in enumerate(Graphs_Train):
        if MDH:
            y_pred = None
        else:
            data = data.to(device)
            y_pred = torch.exp(Models[i](data)).T[1]
        
        
        val = len(FindMinimumTarget(graphs[ig], out = y_pred, threshold = 0.5)[0]) / graphs[ig].number_of_nodes()
        
        value += val
    
    return value / len(Graphs_Train)


# In[43]:


print(f"\nMDH value: {Func('_', MDH = True)}\n")

#for i in range(len(layers)):
for i in range(1):
    
    print(f"\n -- Next layer {layers[i]} -- \n")

    varbound = np.array([[-10,10]] * getDimParams(Models[i]) )

    algorithm_param = {'max_num_iteration' : max_iterations,                       'population_size' : pop_size,                       'mutation_probability' : 0.4,                       'elit_ratio': elitratio,                       'crossover_probability': 0.5,                       'parents_portion': 0.3,                       'crossover_type' : 'uniform',                       'max_iteration_without_improv' : max_iterations//2}

    GA_model = ga(function = Func,                dimension = getDimParams(Models[i]),                variable_type = 'real',                variable_boundaries = varbound,                algorithm_parameters = algorithm_param,
                function_timeout = 1_000_000,
                convergence_curve = False)

    GA_model.run()
    
    sd = getStateDict(Models[i], GA_model.best_variable)
    Models[i].load_state_dict(sd)
    
    
    torch.save(Models[i].state_dict(), 
                       f=f"{PATH_SAVE_TRAINS}{layers[i]}_seed_{SEED}_thr_{int(threshold*10)}_date_{dt_string}.pt")


# In[ ]:




