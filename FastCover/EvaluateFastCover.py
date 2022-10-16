import sys
sys.path.append("../")  # Change the path.
import os
import time
from pathlib import Path
import logging

from glob import glob

import torch
import pandas as pd

import numpy as np


records = []
repeat = 3
ts = np.zeros(repeat)
for i in range(repeat):
    # Select seeds
    t_start = time.time()
    ## =================================================================================================
    # Este es el output de la red, en donde se tienen las probabilidades de que cada nodo sea parte de la
    # solución final
    ## =================================================================================================
    out = net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(1)#!!!!!!!!!!!!!!
    ## =================================================================================================
    # Esta función regresas el k nodos con la máxima influencia
    _, nn_seeds = torch.topk(out, k)
    ## =================================================================================================
    ts[i] = (time.time() - t_start)

# Evaluate time
t_mean = ts.mean() 
t_std = ts.std() / np.sqrt(repeat)

## =================================================================================================
# Nodos cubiertos del total, para nosotros d = 1 ya que solo se puede llegar al siguiente
# baselines.heuristics.py -> get_influence_d
# 
## =================================================================================================
n_covered = get_influence(graph, nn_seeds)
n, m = graph.vcount(), graph.ecount()
## =================================================================================================
print(f"k: {k}. Coverage: {n_covered}/{n}={n_covered/n:.2f}. Time: {t_mean:.2f} ({t_std:.2f})")
model_name = "GRAT3"
# Write to records
records.append({
    "graph": graph_name,
    "model": model_name,
    "seed": seed,
    "n": n,
    "m": m,
    "d": d,
    "k": k,
    "n_covered": n_covered,
    "coverage": n_covered/n,
    "t_mean": t_mean,
    "t_std": t_std,
})