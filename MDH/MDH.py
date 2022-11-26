import igraph
import dgl
import torch
import time
import os
import sys
sys.path.append("./FastCover/")
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

parser.add_argument("-th", "--Threshold", help = "Infection Threshold", type = float)
parser.add_argument("-type", "--Type", help = "short, large or full", type = str)
args = parser.parse_args()

# Example: python EvaluateFastCover.py -th 0.5 -type "short"

Graphs_short = [
    'graph_football.txt',
    'graph_jazz.txt',
    'graph_karate.txt',
    'graph_CA-CondMat.txt',
    'gemsec_facebook_artist.txt',
    'ego-facebook.txt',

 'graph_actors_dat.txt',
 'graph_CA-AstroPh.txt',
 'graph_CA-CondMat.txt',
 'graph_CA-GrQc.txt',
 'graph_CA-HepPh.txt',
 'graph_CA-HepTh.txt',
 'graph_dolphins.txt',
 'graph_Email-Enron.txt',
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

PATH_TO_TEST = "./BRKGA/instances/txt/"

if args.Type == "short":
    Graphs = Graphs_short
elif args.Type == "large":
    Graphs = Graphs_large
elif args.Type == "full":
    Graphs = [graph for graph in os.listdir(PATH_TO_TEST)]
else:
    raise NameError("Only: 'short', 'large' or 'full")

PATH_SAVE_RESULTS = ''

NAME_SAVE_RESULTS = 'MDH' #Change this

directed_test = False

threshold = args.Threshold
dt_string = datetime.now().strftime("%m-%d_%H-%M")

records = []

Total = len(Graphs)
    
print()
print(f"Evaluation of MDH")
print()
c = 1
for file in Graphs:
        print(f"Loading {PATH_TO_TEST+file} ...")
        name = file.split(".")[0].replace("graph_", "")

        graph = igraph.Graph().Read_Edgelist(PATH_TO_TEST + file)
        
        print("\nStarting infection\n")

        G = graph.to_networkx().to_undirected()
        n = len(G.nodes())

        start_time = time.time()        
        _ , minTargetGRAT = FindMinimumTarget(G, out = None, threshold = threshold)
        final_time = (time.time() - start_time)

        print(f"{c}/{Total} Graph: {name}")
        print(f"Best Target Set length: {minTargetGRAT} out of {n}")
        print(f"Ratio Solution / Graph lentgh: {minTargetGRAT/n:.3f}")
        print(f"Time: {final_time:.2f}s")
        print()
        records.append({
        "graph": name,
        "model": 'MDH',
        "seed": 'None',
        "threshold": threshold,
        "n_covered": minTargetGRAT,
        "n": n,
        "coverage": minTargetGRAT/n,
        "t_mean": final_time
        })

        pd.DataFrame(records).to_csv(PATH_SAVE_RESULTS + NAME_SAVE_RESULTS +"_" + dt_string + ".csv")
        gc.collect()
        c += 1
        
print(f"Evaluation has finnished successfully. \nData saved in {PATH_SAVE_RESULTS}")


