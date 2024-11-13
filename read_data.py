import torch
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx

edge_list = []
with open('Metting/Montagna_meetings_edgelist.txt', 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 3:
            source, target, _ = map(int, parts)
            edge_list.append((source, target))
G = nx.Graph()
G.add_edges_from(edge_list)
degree_centrality = np.array(list(nx.degree_centrality(G).values()))
closeness_centrality = np.array(list(nx.closeness_centrality(G).values()))
betweenness_centrality = np.array(list(nx.betweenness_centrality(G).values()))
pagerank = np.array(list(nx.pagerank(G).values()))
features = np.stack([degree_centrality, closeness_centrality, betweenness_centrality, pagerank], axis=1)
node_features = torch.tensor(features, dtype=torch.float)
data = from_networkx(G)
data.x = node_features
# edge_index = torch.tensor(list(G.edges)).t()
# data = Data(x=node_features, edge_index=edge_index)
print(data)
# dataset = Planetoid(root='./src/', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]
# print(type(data))
torch.save(data, 'Metting/Metting.pt')