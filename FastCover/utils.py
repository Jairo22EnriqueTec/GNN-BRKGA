import igraph
import dgl
import torch
import numpy as np
import networkx as nx
from create_dataset import CreateDataset
import gc

def size(V):
    return np.dot(V, np.ones(len(V), dtype = "int32"))

def CheckInfect(G, Infected, Num_Neighs, Num_Neighs_Infected, Can_Sum, n, threshold = 0.5):    
    InfectedTemp = np.array([])
    while size(InfectedTemp) != size(Infected):
        InfectedTemp = Infected.copy()

        for Inf in np.where((Infected * Can_Sum) == 1)[0]:
            #print(f"Nodo inf act {Inf}")
            Temp = np.zeros(n, dtype = "int16")
            Temp[np.array(list(nx.neighbors(G, Inf)))] = 1

            # A cada vecino del nodo infectado actual se le suma 1 si están conectados a este nodo
            Num_Neighs_Infected += Temp
            # El nodo infectado actual ya no puede sumar puesto ya ha sumado a todos sus vecinos un nodo infectado
            Can_Sum[Inf] = 0
            # si el número el ratio de vecinos infectados supera el umbral, dichos nodos se infectan, si no, no suma nada

            Infected += (Num_Neighs_Infected/Num_Neighs >= threshold)

            Infected[Infected>1]=1
    return size(Infected) == n, size(Infected)/n#, Infected, Can_Sum


def FindMinimumTarget(G, out = None, threshold = 0.5):
    """
    in:
    G - networkx
    out - probabilities from torch
    threshold = 0.5 - umbral de infección
    if out = None --> MDH
    """
    
    Solution = []#list(nx.isolates(G))
    n = len(G.nodes())
    Isolates = list(nx.isolates(G))
    
    Num_Neighs = np.array(nx.degree(G)).T[1]
    Num_Neighs_Infected = np.zeros(n, dtype = "int16")
    Can_Sum = np.ones(n, dtype = "int16")
    
    Can_Sum[Isolates] = 0
    
    Infected = np.zeros(n, dtype = "int16")
    Infected[Isolates] = 1
    
    if out == None:
        out_ = Num_Neighs.copy()
    else:
        out_ = out.detach().numpy().copy()
    
    Order_Node_Degree = np.argsort(-out_)
    
    for i in range(n):
        Inf = Order_Node_Degree[i]

        if Infected[Inf] == 1:
            continue
        
        Solution.append(Inf)
        Infected[Inf] = 1
        
        # Modifica todas los vectores Infected, Num_Neighs_Infected y Can_Sum por su ubicación en memoria
        Sol, P = CheckInfect(G, Infected, Num_Neighs, Num_Neighs_Infected, Can_Sum, n,  threshold = threshold)
        
        if Sol:
            break
        if i % (n//20) == 0:
            print(f"{P:.2f} Infected")
    print(f"1.00 Infected")
    print()
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