import igraph
import dgl
import torch
import time
import os
import networkx as nx
from datetime import datetime
import pandas as pd
import sys
sys.path.append("../FastCover/")
from utils import *
import warnings
warnings.filterwarnings('ignore')
import argparse
from models import GNN

parser = argparse.ArgumentParser()

parser.add_argument("-pi", "--PATH", help = "Path to instances", type = str)
parser.add_argument("-ps", "--PATH_Save", help = "Path to save results", type = str)
parser.add_argument("-pm", "--PATH_Model", help = "Path where the models to extract probs are", type = str)
parser.add_argument("-MDH", "--MDH", help = "", type = bool)

args = parser.parse_args()

#v.g. python ExtractProbabilitiesModels.py -pm "runs/Erdos/" -pi "../BRKGA/instances/Erdos/test/" -ps "./probabilidades/Erdos_Erdos/"

PATH_TO_TEST = args.PATH

MDH = bool(args.MDH)

PATH_SAVE_RESULTS = args.PATH_Save

PATH_SAVED_TRAINS = args.PATH_Model


#PATH_TO_TEST = "../BRKGA/instances/txt/"
#PATH_SAVED_TRAINS = "runs/scalefree/"
#PATH_SAVE_RESULTS = 'probabilidades/scalefree/'

Graphs = [graph for graph in os.listdir(PATH_TO_TEST + "txt") if ".txt" in graph]

NAME_SAVE_RESULTS = 'Models' #Change this

Features = [None]*len(Graphs)

if MDH:
    num_features = 1
else:
    num_features = 5 # Change if needed

    #graphFeatures = [feat for feat in os.listdir(PATH_TO_TEST+'feats') if ".txt" in feat]
    Features = []
    for er in Graphs:
        temp = []
        #with open(PATH_TO_TEST+'feats/'+er) as f:
        with open(PATH_TO_TEST+'feats/'+er.replace(".txt", "_feat.txt")) as f:
            for line in f.readlines()[1:]:
                feats = np.array(line.split(","), dtype = float)
                temp.append(feats)
        Features.append(np.array(temp))

        



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_cuda = False

dt_string = datetime.now().strftime("%m-%d_%H-%M")

RUNS_LIST = [run for run in os.listdir(PATH_SAVED_TRAINS) if ".pt" in run]

SEEDS = []
MODELS = []
for run_name in RUNS_LIST:
    SEEDS.append(run_name.split("_")[3])
    MODELS.append(run_name.split("_")[0])
    #EPOCHS.append(run_name.split("_")[2])
    
records = []
Total = len(Graphs)

def save(name, model, out):
    with open(f'{PATH_SAVE_RESULTS}{model}_{name}', 'w') as f:
        out = out.detach().numpy()
        e = 0.0001
        for o in out:
            f.write(str(np.round(o + e, 6)))
            f.write("\n")

    
for run_name, model, seed in zip(RUNS_LIST, MODELS, SEEDS):
    print()
    print(f"Evaluation of model: {model}, seed: {seed} in {run_name}")
    print()
    
    #net = GNN(c_in = num_features, c_hidden = 100, c_out = 2, num_layers = 2, layer_name = model, dp_rate=0.1)
    net = GNN(num_node_features = 5, num_classes = 2, name_layer = model)
    
    net.load_state_dict(torch.load(PATH_SAVED_TRAINS+run_name))
    
    if use_cuda:
        net.cuda()

    c = 1
    for file, feat in zip(Graphs, Features):
            print(f"Loading {PATH_TO_TEST}txt/{file} ...")
            name = file.split(".")[0].replace("graph_", "")

            graph = igraph.Graph.Read_Edgelist(PATH_TO_TEST +"txt/"+ file, directed = False).to_networkx()
            data = Convert2DataSet([graph], [[]], [feat])[0]
            

            #G = graph.to_networkx().to_undirected()

            n = len(graph.nodes())
            
            start_time = time.time()

            # Puesto viene de un log softmax y queremos extraer las probs de que pertenezcan 
            # a la solución, lo cual está en la columna 1
            out = torch.exp(net(data)).T[1]
            
            save(file, model, out)
         
           
            c+=1
print(f"Evaluation has finnished successfully. \nData saved in {PATH_SAVE_RESULTS}")

