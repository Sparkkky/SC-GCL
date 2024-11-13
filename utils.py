import networkx as nx
import numpy as np
import torch
from numpy.linalg import inv, pinv
from torch_geometric.datasets import Planetoid
import scipy.sparse as sp
def compute_D2(B):
    B_rowsum = np.abs(B).sum(axis=1)
    D2 = np.diag(np.maximum(B_rowsum, 1))
    return D2

def compute_D1(B1, D2):
    rowsum = (np.abs(B1) @ D2).sum(axis=1)
    D1 = 2 * np.diag(rowsum)
    return D1
def compute_bunch_matrices(B1, B2):
    D2_2 = compute_D2(B2)
    # D2_1 = compute_D2(B1)
    # D3_n = np.identity(B1.shape[1])
    D1 = compute_D1(B1, D2_2)
    D3 = np.identity(B2.shape[1]) / 3
    D1_pinv = pinv(D1)
    D2_2_inv = inv(D2_2)
    L0u = B1.T @ B1
    L1u = D2_2 @ B1.T @ D1_pinv @ B1
    L1d = B2 @ D3 @ B2.T @ D2_2_inv
    L1f = L1u + L1d
    return L0u, L1f

def get_faces(G):
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            if e1[0] == e2[0]:
                shared = e1[0]
                e3 = (e1[1], e2[1])
            elif e1[1] == e2[0]:
                shared = e1[1]
                e3 = (e1[0], e2[1])
            elif e1[0] == e2[1]:
                shared = e1[0]
                e3 = (e1[1], e2[0])
            elif e1[1] == e2[1]:
                shared = e1[1]
                e3 = (e1[0], e2[0])
            else:
                continue
            if e3[0] in G[e3[1]]:
                faces.append(tuple(sorted((shared, *e3))))
    return list(sorted(set(faces)))
def incidence_matrices(G, V, E, faces, edge_to_idx):
    B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E),len(faces)])

    for f_idx, face in enumerate(faces): # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2
def compute_hodge_matrix(data, sample_data_edge_index):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(data.x.shape[0])])
    edge_index_ = np.array((sample_data_edge_index))
    # edge_index_ = data.edge_index
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                        range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)
    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}
    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx)
    return B1, B2
def get_adj_split(adj, val_prop=0.05, test_prop=0.1):
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)
    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false
def get_edges_split(data, val_prop = 0.2, test_prop = 0.2):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(data.x.shape[0])])
    _edge_index_ = np.array((data.edge_index))
    edge_index_ = [(_edge_index_[0, i], _edge_index_[1, i]) for i in
                        range(np.shape(_edge_index_)[1])]
    g.add_edges_from(edge_index_)
    adj = nx.adjacency_matrix(g)

    return get_adj_split(adj,val_prop = val_prop, test_prop = test_prop)


