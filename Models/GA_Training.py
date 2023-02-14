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
from OwnGenetical import geneticalgorithm as ga
import sys


import pandas as pd
import torch_geometric.transforms as T

from models import GNN

sys.path.append("../FastCover/")

from utils import *


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


# Ejemplo

# python GA_Training.py -pi "../BRKGA/instances/scalefree/train/" -ps "./runs/scalefree/GA/" -s 13 -i 100 -pop 20 -elit 0.01

threshold = 0.5

dt_string = datetime.now().strftime("%m-%d_%H-%M")

# En primer lugar se cargarán las instancias scale-free que son las que se toman para el entrenamiento.

def CargarDataset(PATH_TO_TRAIN):
    
    Instances = [graph for graph in os.listdir(PATH_TO_TRAIN + 'txt')]
    Instances.sort()

    graphs = []
    for er in Instances:
        graph = igraph.Graph.Read_Edgelist(PATH_TO_TRAIN+"txt/"+er, directed = False)
        graphs.append(graph.to_networkx())    


    # Se cargan las features de cada uno de los grafos, /feats tiene las features para cada uno de los grafos.
    # El formato de nombre para los grafos es: "graph1__2.25_1000_30.txt" y para las features: "graph1__2.25_1000_30_feat.txt"
    print("\nCargando Features...\n")
    graphFeatures = [feat for feat in os.listdir(PATH_TO_TRAIN+'feats')]
    graphFeatures.sort()

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
            Features.append(temp)
        except:
            print(er)
            print(line)
            print(c)

    # Convert2DataSet recibe: set de grafos de entreno, soluciones óptimas o None, features. 
    return Convert2DataSet(graphs, None, feats = Features), graphs

## ========================== SCALE FREE =====================================
# El directorio PATH_TO_TRAIN es el directorio dentro del cual /txt contiene todos los grafos scalefree a utilizar 
Graphs_Train, graphs = CargarDataset(PATH_TO_TRAIN)
num_features = Graphs_Train[0].num_features
num_classes = Graphs_Train[0].num_classes


# Se hace el mismo proceso para los grafos de Erdos o de validación
# ======================== ERDOS =================
PATH_TO_TRAIN_er = "../BRKGA/instances/Erdos/train/"

Graphs_Train_Erdos, graphs_er = CargarDataset(PATH_TO_TRAIN_er)
# ========================

layers = ["SAGE", "GCN", "SGConv", "GAT","GraphConv"]

torch.manual_seed(SEED)
Models = [GNN(num_features, num_classes, name_layer = layer_name) for 
         layer_name in layers]

# ============== Funciones para el entreno ==============



def getDimParams(model):
    # Returns the number of parameters needed for a specific model
    
    sum_ = 0
    for name, param in model.named_parameters():
        m = 1
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
        
        sd[name] = torch.tensor(params[from_:from_+m].reshape(param.shape))
        from_ += m
        
    return sd


def SimpleweightedCrossEntropy(y, p, w):
    return np.sum(y*(1-p)*w[0] + (1-y)*p*w[1]) / len(y)

def Func(X, MDH = False, alpha = 0.7):
    # Objective function
    
    if not MDH:
        sd = getStateDict(Models[i], X)
        Models[i].load_state_dict(sd)
    else:
        alpha = 1
        
    value = 0.0
    loss = 0.0
    
    for ig, data in enumerate(Graphs_Train):
        
        if MDH:
            y_pred = None
        else:
            # Cuando se evalua un indivuduo se aplica el modelo sobre cada grafo 
            # para extraer las probabilidades
            data = data.to(device)
            y_pred = torch.exp(Models[i](data)).T[1]
        
        # Con la función FindMinimumTarget se obtiene la longitud del targetset
        ts = len(FindMinimumTarget(graphs[ig], out = y_pred, threshold = 0.5)[0])
        # se obtiene el ratio de tamaño de la solución / tamaño del nodo, y se suma
        val = ts / graphs[ig].number_of_nodes()
        
        value += val
        
        
        if not MDH:
            # Se inicialia un vector de zeros del tamaño del grafo
            zeros = np.zeros(data.num_nodes)
            # Se asigna un 1 a aquellos nodos que forman parte del targetset
            zeros[torch.topk(y_pred, ts)[1]] = 1
            # Se obtiene el peso de la clase minoritaria como la inverse frequency
            weigth_minoritaria = np.sum(zeros==0)/np.sum(zeros)
            # Se calcula el costo, es decir qué tan lejos está la solución brindada de la óptima.
            loss += SimpleweightedCrossEntropy(zeros, y_pred.detach().numpy(), [weigth_minoritaria, 1])
    
    value /= len(Graphs_Train) 
    loss /= len(Graphs_Train) 
    
    # Se pondera el 70% para el targetset ratio y un 30% para el crossentropy
    return value * (alpha) + loss * (1 - alpha)

def Func2(X, MDH = False, alpha = 0.7, scalefree = True):
    # Misma función que arriba, pero para los grafos de Erdos de validación
    # Es necesaria porque el GA solo recibe la función, no los parámetros de la misma
    # TODO: integrarlas en una misma
    
    if not MDH:
        sd = getStateDict(Models[i], X)
        Models[i].load_state_dict(sd)
    else:
        alpha = 1
        
    value = 0.0
    loss = 0.0
       
    
    for ig, data in enumerate(Graphs_Train_Erdos):
        
        if MDH:
            y_pred = None
        else:
            data = data.to(device)
            y_pred = torch.exp(Models[i](data)).T[1]
        
        
        ts = len(FindMinimumTarget(graphs_er[ig], out = y_pred, threshold = 0.5)[0])
        
        val = ts / graphs_er[ig].number_of_nodes()
        
        value += val
        
        if not MDH:
            zeros = np.zeros(data.num_nodes)
            zeros[torch.topk(y_pred, ts)[1]] = 1
            weigth_minoritaria = np.sum(zeros==0)/np.sum(zeros)
            loss += SimpleweightedCrossEntropy(zeros, y_pred.detach().numpy(), [weigth_minoritaria, 1])
    
    value /= len(Graphs_Train_Erdos) 
    loss /= len(Graphs_Train_Erdos) 
    
    return value * (alpha) + loss * (1 - alpha)


print(f"\nMDH value: {Func('_', MDH = True)}\n")

for i in range(len(layers)):
#for i in range(1):
    
    print(f"\n -- Next layer {layers[i]} -- \n")
    
    # Para cada modelo, se establece el límite de cada valor
    varbound = np.array([[-10,10]] * getDimParams( Models[i]) )
    
    # Se inicializan los parámetros del algoritmo genético
    algorithm_param = {'max_num_iteration' : max_iterations,
                       'population_size' : pop_size,                       
                       'mutation_probability' : 0.4,                       
                       'elit_ratio': elitratio,                       
                       'crossover_probability': 0.5,                       
                       'parents_portion': 0.3,                       
                       'crossover_type' : 'uniform',                       
                       'max_iteration_without_improv' : max_iterations//2}
    
    # se correo el modelo
    GA_model = ga(function = Func,
                  secondfunc = Func2,
                  dimension = getDimParams(Models[i]),                
                  variable_type = 'real',                
                  variable_boundaries = varbound,                
                  algorithm_parameters = algorithm_param,
                  function_timeout = 1_000_000,
                  convergence_curve = False, 
                  name = layers[i], 
                  ps = PATH_SAVE_TRAINS)
    
    # NOTA: las curvas de aprendizaje se generan y se guardan dentro de la función "ga"
    # es importante brindar el name que irá arriba del gráfico y el "ps" path to save 
    # para guardar los parámetros en cada iteración.
    
    GA_model.run()
    
    # se extraé el mejor individuo al final del GA y se carga dentro del modelo
    sd = getStateDict(Models[i], GA_model.best_variable)
    Models[i].load_state_dict(sd)
    
    
    
    # Finalmente se guarda en state_dict del mejor.
    torch.save(Models[i].state_dict(), 
                       f=f"{PATH_SAVE_TRAINS}{layers[i]}_seed_{SEED}_thr_{int(threshold*10)}_date_{dt_string}.pt")


