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
from GRAT import GRAT3
import warnings
warnings.filterwarnings('ignore')
import argparse
from models import GNNModel

parser = argparse.ArgumentParser()

parser.add_argument("-th", "--Threshold", help = "Infection Threshold", type = float)
parser.add_argument("-type", "--Type", help = "short, large or full", type = str)
args = parser.parse_args()
# Example: python EvaluateFastCover.py -th 0.5 -type "short"

Graphs_short = [
 'ego-facebook.txt',
 'gemsec_facebook_artist.txt',
 'graph_actors_dat.txt',
 'graph_CA-AstroPh.txt',
 'graph_CA-CondMat.txt',
 'graph_CA-GrQc.txt',
 'graph_CA-HepPh.txt',
 'graph_CA-HepTh.txt',
 'graph_dolphins.txt',
 'graph_Email-Enron.txt',
 'graph_football.txt',
 'graph_jazz.txt',
 'graph_karate.txt',
 'graph_ncstrlwg2.txt',
 'soc-gplus.txt',
 'socfb-Brandeis99.txt',
 'socfb-Mich67.txt',
 'socfb-nips-ego.txt']

Graphs_large = ['Amazon0302.txt',
 'Amazon0312.txt',
 'Amazon0505.txt',
 'Amazon0601.txt',
 'com-youtube.ungraph.txt',
 'com-dblp.ungraph.txt',
 'loc-gowalla_edges.txt',
 'deezer_HR.txt',
 'musae_git.txt']

PATH_TO_TEST = "../BRKGA/instances/txt/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.Type == "short":
    Graphs = Graphs_short
elif args.Type == "large":
    Graphs = Graphs_large
elif args.Type == "full":
    Graphs = [graph for graph in os.listdir(PATH_TO_TEST)]
else:
    raise NameError("Only: 'short', 'large' or 'full")

PATH_SAVED_TRAINS = "runs/"
PATH_SAVE_RESULTS = 'results/'

NAME_SAVE_RESULTS = 'Models' #Change this

use_cuda = False

threshold = args.Threshold
dt_string = datetime.now().strftime("%m-%d_%H-%M")

RUNS_LIST = [run for run in os.listdir(PATH_SAVED_TRAINS) if ".pt" in run]

SEEDS = []
MODELS = []
for run_name in RUNS_LIST:
    SEEDS.append(run_name.split("_")[2])
    MODELS.append(run_name.split("_")[0])
    
records = []
Total = len(Graphs)
    
for run_name, model, seed in zip(RUNS_LIST, MODELS, SEEDS):
    print()
    print(f"Evaluation of model: {model}, seed: {seed} in {run_name}")
    print()
    
    net = GNNModel(c_in = 1, c_hidden = 100, c_out = 2, num_layers = 2, layer_name = model, dp_rate=0.1)
    net.load_state_dict(torch.load(PATH_SAVED_TRAINS+run_name))
    
    if use_cuda:
        net.cuda()

    c = 1
    for file in Graphs:
            print(f"Loading {PATH_TO_TEST+file} ...")
            name = file.split(".")[0].replace("graph_", "")

            graph = igraph.Graph().Read_Edgelist(PATH_TO_TEST + file)
            data = Convert2DataSet([graph.to_networkx()], [[]])[0]
            
            print("\nStarting infection\n")

            G = graph.to_networkx().to_undirected()

            n = len(G.nodes())
            
            start_time = time.time()

            out = net(data.x, data.edge_index).max(1)[0]
         
            _ , minTargetGRAT = FindMinimumTarget(G, out, threshold)

            final_time = (time.time() - start_time)
            
            print()
            print(f"{c}/{Total} Graph: {name}")
            print(f"Best Target Set length: {minTargetGRAT} out of {n}")
            print(f"Ratio Solution / Graph lentgh: {minTargetGRAT/n:.3f}")
            print(f"Time: {final_time:.2f}s")
            print()
            records.append({
            "graph": name,
            "model": model,
            "seed": seed,
            "threshold": threshold,
            "n_covered": minTargetGRAT,
            "n": n,
            "coverage": minTargetGRAT/n,
            "t_mean": final_time
            })

            pd.DataFrame(records).to_csv(PATH_SAVE_RESULTS + NAME_SAVE_RESULTS +"_" + dt_string + ".csv")

            c+=1
print(f"Evaluation has finnished successfully. \nData saved in {PATH_SAVE_RESULTS}")

