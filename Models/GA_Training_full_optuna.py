#!/usr/bin/env python
# coding: utf-8

# In[10]:


import networkx as nx
import numpy as np
import torch
from datetime import datetime
import os
import optuna
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn as geom_nn
from OwnGenetical import geneticalgorithm as ga
import sys
import subprocess


import pandas as pd
import torch_geometric.transforms as T

from models import GNN

sys.path.append("../FastCover/")

from utils import *

import warnings
warnings.filterwarnings("ignore")


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

# python GA_Training_full.py -pi "../BRKGA/instances/scalefree/train/" -ps "./runs/eliminar/" -s 13 -i 100 -pop 3 -elit 0.01

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

cant_layers = [3]*5
cant_layers += [2]*5
cant_layers += [1]*5

Models = [GNN(num_features, num_classes, name_layer = layer_name, num_layers = 3) for 
         layer_name in layers]

Models += [GNN(num_features, num_classes, name_layer = layer_name, num_layers = 2) for 
         layer_name in layers]

Models += [GNN(num_features, num_classes, name_layer = layer_name, num_layers = 1) for 
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
    # X: 
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

#mutations = [0.1, 0.2, 0.3, 0.4]
#cross_overs = [0.6, 0.5, 0.4, 0.3]
#crossover_types = ["one_point", "two_point", "uniform"]
#Max_iterations = [10, 4]

#Max_iterations = [100, 200, 300]

i = 0

def objective(trial):
    
    global i

    crossover_type = trial.suggest_categorical('crossover_type', ["one_point", "two_point", "uniform"])
    mutation = trial.suggest_float('mutation', 0.1, 0.5)
    Max_iteration = trial.suggest_categorical('Max_iteration', [100, 200, 300])
    cross_over = trial.suggest_float('cross_over', 0.3, 0.6)
    i = trial.suggest_categorical('model', np.arange(len(Models)) )
    
    dir_name = f"{PATH_SAVE_TRAINS}{layers[i%5]}_{cant_layers[i]}_mut_{mutation:.1f}_cross_{cross_over:.1f}_type_{crossover_type}_iter_{Max_iteration}/"
                    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    print(f"\n -- Next layer {layers[i%5]} {cant_layers[i]} -- \n")

    # Para cada modelo, se establece el límite de cada valor
    varbound = np.array([[-10,10]] * getDimParams( Models[i]) )

    # Se inicializan los parámetros del algoritmo genético
    algorithm_param = {'max_num_iteration' : Max_iteration,
                       'population_size' : pop_size,                       
                       'mutation_probability' : mutation, 
                       'elit_ratio': elitratio,                       
                       'crossover_probability': cross_over,                       
                       'parents_portion': 0.3,                       
                       'crossover_type' : crossover_type,                       
                       'max_iteration_without_improv' : Max_iteration//2}

    # se correo el modelo
    GA_model = ga(function = Func,
                  secondfunc = Func2,
                  dimension = getDimParams(Models[i]),                
                  variable_type = 'real',                
                  variable_boundaries = varbound,                
                  algorithm_parameters = algorithm_param,
                  function_timeout = 1_000_000,
                  convergence_curve = False, 
                  name = layers[i%5], 
                  ps = dir_name)

    # NOTA: las curvas de aprendizaje se generan y se guardan dentro de la función "ga"
    # es importante brindar el name que irá arriba del gráfico y el "ps" path to save 
    # para guardar los parámetros en cada iteración.

    GA_model.run()


    with open(dir_name + "LearningCurves.npy", 'rb') as f:
            scf = np.load(f, allow_pickle = True)
            er = np.load(f, allow_pickle = True)

    best_index = np.argmin((scf + er)/2)
    
    best_value = np.min((scf + er)/2)

    with open(f"{dir_name}{layers[i%5]}_iter_{best_index}.npy", 'rb') as f:
            X = np.load(f, allow_pickle = True)

    print(f"El mejor index es la iteración: {best_index}")
    # se extraé el mejor individuo al final del GA y se carga dentro del modelo
    sd = getStateDict(Models[i], X)
    Models[i].load_state_dict(sd)

    r1 = Func(X)
    r2 = Func2(X)

    with open(dir_name+'Res.npy', 'wb') as f:
        np.save(f, np.array([r1, r2]), allow_pickle = True)

    # Finalmente se guarda en state_dict del mejor.

    torch.save(Models[i].state_dict(), 
                       f=f"{dir_name}{layers[i%5]}{cant_layers[i]}_seed_{SEED}_thr_{int(threshold*10)}_date_{dt_string}.pt")

    if not os.path.exists(dir_name + "probs/"):
        os.mkdir(dir_name + "probs/")

    # Se extraen las probabilidades con el mejor modelos guardado.
    res = subprocess.run([
        sys.executable, "ExtractProbabilitiesModels.py", "-pm", dir_name, "-pi",
        "../BRKGA/instances/socialnetworks/", "-ps", dir_name + "probs/"])

    if res.returncode == 1:
        print("No se pudieron cargar las probabilidades")
    elif res.returncode == 0:
        print("Se guardaron las probabilidades")

    #path_save = "runs/eliminar/SAGE_3_mut_0.1_cross_0.6_type_one_point_iter_3/Results.txt"

    #path_prob = "runs/eliminar/SAGE_3_mut_0.1_cross_0.6_type_one_point_iter_3/probs/";

    # Se corre el algoritmo de difusión en C++

    res = subprocess.run(["./diffusion.exe", "-pp", dir_name + "probs/", "-ps", 
                          dir_name + "Res_diffusion.txt", "-m", layers[i%5]])

    if res.returncode == 1:
        print("No se pudo correr el algoritmo de difusión.")
    elif res.returncode == 0:
        print("Los resultados están listos.")

    DF = pd.read_csv(dir_name + "Res_diffusion.txt", header = None)
    
    
    return DF.iloc[:,1].mean()


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 200)

