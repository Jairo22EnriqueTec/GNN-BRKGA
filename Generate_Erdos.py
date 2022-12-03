import networkx as nx
import numpy as np
import progressbar
import igraph
PATH_INSTANCES_TXT = "./BRKGA/instances/Erdos/txt/"
PATH_INSTANCES_DIMACS = "./BRKGA/instances/Erdos/dimacs/"

def nx2dimacs(g, n, PATH_INSTANCES):
    dimacs_filename = f"{PATH_INSTANCES}ER_{n}_20_{Cont}.dimacs"

    with open(dimacs_filename, "w") as f:
        # write the header
        f.write("p edge {} {}\n".format(g.number_of_nodes(), g.number_of_edges()))
        # now write all edges
        for u, v in g.edges():
            f.write("e {} {}\n".format(u+1, v+1))

def nx2txt(G, n, PATH_INSTANCES):
    file = open(f"{PATH_INSTANCES}ER_{n}_20_{Cont}.txt", 'w')
    c = 0
    for i,j in G.edges():
        file.write(f"{int(i)} {int(j)}")
        file.write('\n')
        c += 1
    file.close()


NumGrafos = [4, 4, 4, 2, 2, 1, 1]
lens = [1_000, 2_000, 5_000, 10_000, 20_000, 30_000, 50_000]
INSTANCES = []

for k in progressbar.progressbar(range(len(NumGrafos))):
    Cont = 0
    n = lens[k]
    p = 20/n 
    
    while True:
        if Cont == NumGrafos[k]:
            break
        G = nx.erdos_renyi_graph(n, p, directed = False)
        AllConected = True if list(nx.isolates(G)) == [] else False
        
        if AllConected:
            nx2txt(G, n, PATH_INSTANCES_TXT)
            nx2dimacs(G, n, PATH_INSTANCES_DIMACS)
            Cont += 1
            print(f"\n{Cont}/{NumGrafos[k]}")
            
        

            