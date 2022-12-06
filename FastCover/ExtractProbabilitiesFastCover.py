#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import igraph
import dgl
import torch
import time
import os
import networkx as nx
from datetime import datetime
import pandas as pd
from utils import *
from GRAT import GRAT3
import warnings
warnings.filterwarnings('ignore')
import argparse
import gc


parser = argparse.ArgumentParser()

parser.add_argument("-pi", "--PATH", help = "Path to instances", type = str)
parser.add_argument("-ps", "--PATH_Save", help = "Path to save results", type = str)
parser.add_argument("-pm", "--PATH_Model", help = "Path where the models to extract probs are", type = str)

args = parser.parse_args()

#v.g. python ExtractProbabilitiesFastCover.py -pm "./runs/scalefree/" -pi "../BRKGA/instances/Erdos/test/txt/" -ps "./probabilidades/scalefree_Erdos/"


PATH_TO_TEST = args.PATH

PATH_SAVE_RESULTS = args.PATH_Save

PATH_SAVED_TRAINS = args.PATH_Model

#PATH_TO_TEST = "../BRKGA/instances/txt/"

Graphs = [graph for graph in os.listdir(PATH_TO_TEST)]


#PATH_SAVE_TRAINS = "runs/scalefree/"
#PATH_SAVE_RESULTS = 'probabilidades/scalefree/'

NAME_SAVE_RESULTS = 'FastCover' #Change this

FEATURE_TYPE = "1"
HIDDEN_FEATS = [32]*6
input_dim = 32
use_cuda = False
directed_test = False

dt_string = datetime.now().strftime("%m-%d_%H-%M")

RUNS_LIST = [run for run in os.listdir(PATH_SAVED_TRAINS) if ".pt" in run]

SEEDS = []
MODELS = []
for run_name in RUNS_LIST:
    SEEDS.append(run_name.split("_")[2])
    MODELS.append(run_name.split("_")[0])


records = []

Total = len(Graphs)

def save(name, out):
    with open(f'{PATH_SAVE_RESULTS}FC_{name}', 'w') as f:
        out = out.detach().numpy()
        e = 0.0001
        for o in out:
            f.write(str(np.round(o+e, 6)))
            f.write("\n")

    
for run_name, model, seed in zip(RUNS_LIST, MODELS, SEEDS):
    print()
    print(f"Evaluation of model: {model}, seed: {seed} in {run_name}")
    print()
    
    if model == 'GRAT':
        net = GRAT3(*HIDDEN_FEATS)
        net.load_state_dict(torch.load(PATH_SAVED_TRAINS+run_name))
    if use_cuda:
        net.cuda()

    c = 1
    for file in Graphs:
            print(f"Loading {PATH_TO_TEST+file} ...")
            name = file.split(".")[0].replace("graph_", "")

            graph = igraph.Graph().Read_Edgelist(PATH_TO_TEST + file)

            dglgraph = get_rev_dgl(graph, FEATURE_TYPE, input_dim, directed_test, use_cuda)
            
            G = graph.to_networkx().to_undirected()

            n = len(G.nodes())

            start_time = time.time()

            out = torch.sigmoid(net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(1))
            
            save(file, out)
            
            c+=1
            
print(f"Evaluation has finnished successfully. \nData saved in {PATH_SAVE_RESULTS}")


