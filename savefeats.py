#!/usr/bin/env python
# coding: utf-8

# In[152]:


import numpy as np
import networkx as nx
import igraph
import progressbar
from icecream import ic
import os
import time
import matplotlib.pyplot as plt
ic.configureOutput(prefix = 'debug | -> ')
ic.disable()
ic.enable()

import argparse



parser = argparse.ArgumentParser()

parser.add_argument("-p", "--PATH", help = "", type = str)
parser.add_argument("-ps", "--PATH_SAVE", help = "", type=str)

args = parser.parse_args()

PATH = args.PATH
PATH_save = args.PATH_SAVE


#PATH = './BRKGA/instances/Erdos/test/txt/'
#PATH_save = './BRKGA/instances/Erdos/test/feats/'


def getFeatures(G):
    
    BC = np.array(list(nx.betweenness_centrality(G, k = 500).values()))
    CC = np.array(list(nx.closeness_centrality(G).values()))
    LC = np.array(list(nx.load_centrality(G).values()))
    DG = np.array(list(nx.degree(G))).T[1]
    PR = np.array(list(nx.pagerank(G).values()))

    features = [BC, PR, DG, CC, LC]
    names = ["BC", "PageRank", "degree", "closeness_centrality", "LC"]
    return np.array(features).T, names


# In[227]:


def writeFeatures(PATH, ins, features, elapsed):
    subfij = '_feat'
    file2 = open(PATH + ins.split(".")[0] + subfij + ".txt", 'w')
    c = 0
    
    file2.write(f"time: {elapsed}, n: {features.shape[0]}")
    file2.write('\n')
    
    for f in features:
        st = str(f).replace("\n", "").replace("[", "").replace("]", "").replace(" ", ",")
        file2.write(st)
        file2.write('\n')
        c += 1
    file2.close()
    print(f"para {ins} se escribieron {c} lines")


# In[231]:

graphs = [graph for graph in os.listdir(PATH)]

Graphs = []
for ins in graphs:
    G = igraph.Graph.Read_Edgelist(PATH+ins, directed = False)
    G = G.to_networkx()
    Graphs.append(G)


# In[229]:


c = 0
for G, ins in zip(Graphs, graphs):
    c+=1
    print(f"\n------------ {c} out of {len(Graphs)} ------------\n")
    print(f"\nNext graph: {ins}")
    
    s = time.time()
    features, _ = getFeatures(G)
    elapsed = time.time() - s
    print(f"\nTime elapsed: {elapsed:.3f}")
    
    writeFeatures(PATH_save, ins, features, elapsed)
    


# In[204]:
