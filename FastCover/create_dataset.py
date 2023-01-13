from torch_geometric.data import InMemoryDataset, Data
import networkx as nx
import torch
import numpy as np

class CreateDataset(InMemoryDataset):
    def __init__(self, G, labels, transform = None, feats = None, scale = True):
        super(CreateDataset, self).__init__('.', transform, None, None)

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        
        data = Data(edge_index=edge_index)

        # using degree as embedding
        if np.all(feats == None):
            embeddings = np.array(list(dict(G.degree()).values()))
            #embeddings = embeddings/np.max(embeddings)
            # normalizing degree values
            embeddings = embeddings.reshape(-1,1)
        else:
            embeddings = feats
        
        # Se escalan los embeddings para obtener medidas que pueda aprender la red
        if scale:
            embeddings = (embeddings - np.min(embeddings, axis = 0)) /\
                     ( np.max(embeddings, axis = 0) - np.min(embeddings, axis = 0) )
        else:
            embeddings = (embeddings - np.mean(embeddings, axis = 0)) /\
                     np.std(embeddings, axis = 0)
        
        data.num_nodes = G.number_of_nodes()
        
        # embedding 
        data.x = torch.from_numpy(embeddings).type(torch.float32)
        
        # labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach()
        
        data.num_classes = 2

        # splitting the data into train, validation and test
        """
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())), 
                                                            pd.Series(labels),
                                                            test_size=0.30, 
                                                            random_state=42)
        """
        
        n_nodes = G.number_of_nodes()
        # create train and test masks for data
        """
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True
        test_mask[X_test.index] = True
        data['train_mask'] = train_mask
        data['test_mask'] = test_mask
        """
        self.data, self.slices = self.collate([data])

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)