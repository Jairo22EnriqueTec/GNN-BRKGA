import igraph
import dgl
import torch
import numpy as np
import networkx as nx
from create_dataset import CreateDataset
import gc

def size(V):
    return np.dot(V, np.ones(len(V)))

def CheckInfect(G, Infected, AM, Num_Neighs, n, threshold = 0.5):    
   
    InfectedTemp = np.array([])
    #"""
    while size(InfectedTemp) != size(Infected):
        InfectedTemp = Infected.copy()
        New = ((np.array(np.dot(AM, Infected.T))[0] / np.array(Num_Neighs)) > threshold)
        if np.all(New[0] == Infected):
            break
        Infected += New[0]
        Infected[Infected>1] = 1
        
    return size(Infected) == n, size(Infected)/n, Infected

def FindMinimumTarget(G, out = None, threshold = 0.5):
    """
    in:
    G - networkx
    out - probabilities from torch
    threshold = 0.5 - umbral de infección
    if out = None --> MDH
    """
    
    Solution = []
    n = len(G.nodes())

    
    Infected = np.zeros(n, dtype = "int16")
    
    AM = nx.adjacency_matrix(G).todense()
    AM = np.matrix(AM, dtype = "int16")
    
    Num_Neighs = size(AM)
    Num_Neighs = np.array(Num_Neighs, dtype = "int16")
    
    if out == None:
        out_ = Num_Neighs.copy()
    else:
        out_ = out.detach().numpy().copy()
    
    #G = graph.to_networkx()
    for i in range(n):

        Inf = np.argmax(out_)
        out_ = np.delete(out_, Inf)

        if Infected[Inf] == 1:
            continue
        
        Solution.append(Inf)
        Infected[Inf] = 1
        
        Sol, P,  Infected = CheckInfect(G, Infected, AM, Num_Neighs, n,  threshold = threshold)
        #Infected = list(Infected)
        
        if Sol:
            break
        if i % (n//10) == 0:
            print(f"{P:.2f} Infected")
    
    del AM, Num_Neighs
    gc.collect()
    
    return Solution, len(Solution)


def get_rev_dgl(graph, feature_type='0', feature_dim=None, is_directed=False, use_cuda=False):
    """get dgl graph from igraph
    """
    
    src, dst = zip(*graph.get_edgelist())

    if use_cuda:
        dglgraph = dgl.graph((dst, src)).to(torch.device("cuda:0"))
    else:
        dglgraph = dgl.graph((dst, src))
        
    if not is_directed:
        dglgraph.add_edges(src, dst)

    if use_cuda:
        dglgraph.ndata['feat'] = FEATURE_TYPE_DICT[feature_type](graph, feature_dim).cuda()
        dglgraph.ndata['degree'] = torch.tensor(graph.degree()).float().cuda()

    else:
        dglgraph.ndata['feat'] = FEATURE_TYPE_DICT[feature_type](graph, feature_dim)
        dglgraph.ndata['degree'] = torch.tensor(graph.degree()).float()
        
    return dglgraph

def gen_one_feature(graph, feature_dim):
    """Generate all-one features
    """
    return torch.ones(graph.vcount(), feature_dim)


FEATURE_TYPE_DICT = {
    "1": gen_one_feature,
}

def MDH(G, threshold = 0.5, print_ =False):
    """
    Aplica la Heurística del Grado Mínimo para obtener el conjunto 
    más pequeño posible de nodos.
    
    In:
    G - Grafo networkx
    threshold = 0.5 - Umbral de infección
    print_ = False - Mostrar avance de infección
    
    out:
    solution - Conjunto mínimo encontrado de nodos infectados
    """
    
    NodesDegree = np.array(nx.degree(G))
    Infected = [k for k in nx.isolates(G)]
    # Si llega a existir nodos aislados, estos harán que el método sea trivial
    # Por eso los infectados iniciales son los aislados
    Solution = []
    while len(NodesDegree) != 0 and len(Infected) != len(G.nodes()):

        posMaxDegreeNode = np.argmax(NodesDegree.T[1])
        MaxDegreeNode = NodesDegree[posMaxDegreeNode][0]

        NodesDegree = np.delete(NodesDegree, posMaxDegreeNode, axis=0)

        Solution.append(MaxDegreeNode)
        Infected.append(MaxDegreeNode)
            
        InfectedTemp = []
        while len(InfectedTemp) != len(Infected):
            InfectedTemp = Infected.copy()
            for Inf in InfectedTemp:
                for neighborL1 in nx.neighbors(G, Inf):

                    if neighborL1 in Infected:
                        continue

                    TotalNeighbors = [v for v in nx.neighbors(G, neighborL1)]
                    NeighborsInfedted = [v for v in TotalNeighbors if v in Infected]

                    ratio = len(NeighborsInfedted)/len(TotalNeighbors)
                    if ratio > threshold:
                        Infected.append(neighborL1)
                        NodesDegree = np.delete(NodesDegree, np.where(NodesDegree.T[0] == neighborL1)[0], axis = 0)
        if print_:
            print(f"{len(Infected)/len(G.nodes()):.3f} infectado")
    return Solution, len(Infected)

def Convert2DataSet(Graphs, Optimals):
    g = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for G, OptimalSet in zip(Graphs, Optimals):
        NumNodes = G.number_of_nodes()
        labels = np.zeros(NumNodes)
        labels[OptimalSet] = 1

        dataset = CreateDataset(G, labels)
        data = dataset[0]

        data =  data.to(device)
        g.append(data)
        
    return g