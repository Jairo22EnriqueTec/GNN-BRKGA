import igraph
import dgl
import torch
import networkx as nx

def CheckInfect(G, Infected, threshold):
    """
    Recibe Una red G, conjunto de nodos infectados y el umbral
    Propaga la infección en la red todo lo que puede
    Regresa verdadero si se ha infectado toda la red o
    falso en lo contrario
    
    In:
    G - grafo networkx
    Infected - Conjunto de nodos iniciales infectados
    threshold - Umbral de infección
    """
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
    return len(Infected) == len(G.nodes()), len(Infected)/len(G.nodes())

def FindMinimumTarget(G, out, threshold = 0.5):
    """
    in:
    G - networkx
    out - probabilities from torch
    threshold = 0.5 - umbral de infección
    
    """
    #G = graph.to_networkx()
    
    kMax = int(len(G.nodes()) * threshold)
    kMin = 0
    j = []
    for _ in range(10):
        
        
        _, Infected = torch.topk(out, kMax)
        
        # Anexando los nodos isolados, si existen
        Infected = list(Infected.numpy())
        for i in nx.isolates(G):
            Infected.append(i)
        
        
        Inf, _ = CheckInfect(G, Infected, threshold)
        
        
        if Inf:
            kMax = kMax - (kMax - kMin)//2
        else:
            t = kMax
            kMax = kMax + (kMax - kMin)//2
            kMin = t
        if (kMax - kMin) == 1:
            break
    #print(f"El mejor es {kMax}")
    return Infected, kMax

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