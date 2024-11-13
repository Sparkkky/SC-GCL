import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, from_networkx


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()
def load_data_airport(dataset_str, data_path, return_label=True):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    # edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    # if return_label:
    #     label_idx = 4
    #     labels = features[:, label_idx]
    #     features = features[:, :label_idx]
    #     labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
    #     return edge_index, features, labels
    # else:
    #     return edge_index, features
    data = from_networkx(graph)
    data.x = torch.tensor(features, dtype=torch.float)
    return data

data = load_data_airport('airport', '../src/Airports')
# data = Data(x=torch.tensor(features, dtype=torch.float),
#             edge_index=edge_index,
#             y=torch.tensor(labels, dtype=torch.long))
print(data)
torch.save(data, 'Airports/Airports.pt')