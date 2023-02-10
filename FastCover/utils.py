import igraph
import dgl
import torch
import numpy as np
import networkx as nx
from create_dataset import CreateDataset
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import gc

class EarlyStopping():
    def __init__(self, patience, PATH_SAVE_TRAINS, layer, SEED, threshold, epochs):
        self.PATH = PATH_SAVE_TRAINS
        self.layername = layer
        self.seed = SEED
        self.thr = threshold
        
        self.ep = epochs
        
        self.minloss = np.inf
        self.dt_string = datetime.now().strftime("%m-%d_%H-%M")
        self.count = 0
        self.patience = patience
        self.epoch = 0
        self.history = []
        
    def check(self, model, Graphs_Val, Graphs_Train):
        self.epoch += 1
        if Graphs_Val != "":
            
            val_loss = 0
            for data in Graphs_Val:
                val_loss += F.nll_loss(model(data.x, data.edge_index), data.y).detach().numpy() / len(Graphs_Val)

            if val_loss < self.minloss:
                print(f"\nval_loss improved from {self.minloss :.3f} to {val_loss:.3f} Saving...")
                torch.save(model.state_dict(), f=f"{self.PATH}{self.layername}_epochs_{self.ep}_seed_{self.seed}_thr_{int(self.thr*10)}_date_{self.dt_string}.pt")
                self.minloss = val_loss
                self.count = 0
            else:
                print(f"\nEarlyStopping count {self.count} of {self.patience}\n")
                self.count += 1
        
        
            val_train = 0
            for data in Graphs_Train:
                val_train += F.nll_loss(model(data.x, data.edge_index), data.y).detach().numpy() / len(Graphs_Train)

            self.history.append([self.epoch, val_loss, val_train])

            if self.count >= self.patience:
                print("\nEarlyStopping limit reached.\n")
                return True
            else:
                return False
            
        else:
            val_train = 0
            for data in Graphs_Train:
                val_train += F.nll_loss(model(data.x, data.edge_index), data.y).detach().numpy() / len(Graphs_Train)
                
                
            if val_train < self.minloss:
                print(f"\ntrain_loss improved from {self.minloss :.3f} to {val_train:.3f} Saving...")
                torch.save(model.state_dict(), f=f"{self.PATH}{self.layername}_epochs_{self.ep}_seed_{self.seed}_thr_{int(self.thr*10)}_date_{self.dt_string}.pt")
                self.minloss = val_train
                self.count = 0
            else:
                print(f"\nEarlyStopping count {self.count} of {self.patience}\n")
                self.count += 1

            self.history.append([self.epoch, val_train])

            if self.count >= self.patience:
                print("\nEarlyStopping limit reached.\n")
                return True
            else:
                return False
            
        
    
    def plot_history(self, title_ = ""):
        plt.plot(np.array(self.history).T[0], np.array(self.history).T[1], label = "Val loss")
        plt.plot(np.array(self.history).T[0], np.array(self.history).T[2], label = "Train loss")
        plt.title(f"Learning curve {title_}")
        plt.legend()
        plt.grid()
        plt.show()
    

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


def FindMinimumTarget(G, out = None, threshold = 0.5, print_ = False):
    """
    in:
    G - networkx
    out - probabilities from torch
    threshold = 0.5 - umbral de infección
    if out = None --> MDH
    """
    Isolates = list(nx.isolates(G))
    Solution = Isolates
    n = len(G.nodes())
    
    
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
        if i % (n//20) == 0 and print_:
            print(f"{P:.2f} Infected")
    if print_:
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

def Convert2DataSet(Graphs, Optimals = None, feats = None, scale = True):
    g = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if np.all(feats == None):
        feats = [None] * len(Graphs)
    if np.all(Optimals == None):
        Optimals = [0] * len(Graphs)
        
    for G, OptimalSet, feat in zip(Graphs, Optimals, feats):
        NumNodes = G.number_of_nodes()
        labels = np.zeros(NumNodes)
        labels[OptimalSet] = 1

        dataset = CreateDataset(G, labels, feats = feat, scale = scale)
        data = dataset[0]

        data =  data.to(device)
        g.append(data)
        
    return g