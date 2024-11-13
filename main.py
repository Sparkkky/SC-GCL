import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import torch.nn as nn
from utils import compute_hodge_matrix, get_edges_split, compute_bunch_matrices

dataset = Planetoid(root='Cora', name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, L0_zie):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.weights_L_0 = nn.Parameter(torch.FloatTensor(int(L0_zie), 32))
        self.weights_L_1 = nn.Parameter(torch.FloatTensor(int(L0_zie), 32))
        self.weights_off_diagonal = nn.Parameter(torch.FloatTensor(int(L0_zie),
                                                                   int(L0_zie)))
        self.embeddings_sim = nn.Parameter(torch.FloatTensor(data.x.size(1), int(L0_zie*2)))
        self.weights_sim = nn.Parameter(torch.FloatTensor(int(L0_zie*2), 8))
        self.lin = nn.Linear(8, num_classes)


    def Block_Hodge(self, x, L0, L1):
        L0_r = torch.matrix_power(L0, 2)
        L1_r = torch.matrix_power(L1, 2)
        relation_embedded = torch.einsum('xd, dy -> xy', torch.matmul(L0_r, self.weights_L_0),
                                         torch.matmul(L1_r, self.weights_L_1).transpose(0, 1))
        relation_embedded_ = torch.matmul(self.weights_off_diagonal, relation_embedded)
        upper_block = torch.cat([L0_r, relation_embedded_], dim=1)
        lower_block = torch.cat([torch.transpose(relation_embedded_, 0, 1), L1_r], dim=1)
        sim_block = torch.cat([upper_block, lower_block], dim=0)
        sim_block = F.softmax(F.relu(sim_block), dim=1)
        embeddings_sim = torch.matmul(x, self.embeddings_sim)
        s_emb_sim_ = torch.matmul(embeddings_sim, sim_block)
        s_emb_sim = torch.matmul(s_emb_sim_, self.weights_sim)
        s_emb_sim = s_emb_sim.renorm_(2, 0, 1)
        return s_emb_sim
    def forward(self, data, L0u, L1f):
        x, edge_index = data.x, data.edge_index
        s_emb = self.Block_Hodge(x, L0u, L1f)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        s_emb_x = self.lin(s_emb)
        x = x + s_emb_x
        # print(x.shape)
        # x = F.relu(self.conv1(x, edge_index)) + s_emb
        # x = F.dropout(x, training=self.training)
        # x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
# val_prop = 0.05
# test_prop = 0.1
# train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_edges_split(data, val_prop = val_prop, test_prop = test_prop)
# total_edges = np.concatenate((train_edges,train_edges_false,val_edges,val_edges_false,test_edges,test_edges_false))
# data.train_pos,data.train_neg = len(train_edges),len(train_edges_false)
# data.val_pos, data.val_neg = len(val_edges), len(val_edges_false)
# data.test_pos, data.test_neg = len(test_edges), len(test_edges_false)
# data.total_edges = total_edges
# data.total_edges_y = torch.cat((torch.ones(len(train_edges)), torch.zeros(len(train_edges_false)), torch.ones(len(val_edges)), torch.zeros(len(val_edges_false)),torch.ones(len(test_edges)), torch.zeros(len(test_edges_false)))).long()
# edge_list = np.array(data.edge_index).T.tolist()
# for edges in val_edges:
#     edges = edges.tolist()
#     if edges in edge_list:
#         edge_list.remove(edges)
#         edge_list.remove([edges[1], edges[0]])
# for edges in test_edges:
#     edges = edges.tolist()
#     if edges in edge_list:
#         edge_list.remove(edges)
#         edge_list.remove([edges[1], edges[0]])
# data.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)
random_edge_num = 500
indices = np.random.choice((data.edge_index).size(1), (random_edge_num,), replace=False)
indices = np.sort(indices)
sample_data_edge_index = data.edge_index[:, indices]
boundary_matrix0_, boundary_matrix1_ = compute_hodge_matrix(data, sample_data_edge_index)
L0u, L1f = compute_bunch_matrices(boundary_matrix0_, boundary_matrix1_)
L0u = torch.tensor(L0u, dtype=torch.float32)
L1f = torch.tensor(L1f, dtype=torch.float32)
L0_zie = L0u.shape[0]
model = GCN(dataset.num_node_features, dataset.num_classes, L0_zie)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
best_acc = []
for epoch in range(300):
    optimizer.zero_grad()
    out = model(data, L0u, L1f)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    model.eval()
    _, pred = model(data, L0u, L1f).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    accuracy = correct / data.test_mask.sum().item()
    best_acc.append(accuracy)
    print(f'Epoch [{epoch + 1}/{200}], Loss: {round(loss.item(), 4)}, Accuracy: {round(accuracy, 4)}')
print('*************************************************')
print('Best Acc: {:.4f}'.format(max(best_acc)))
print('*************************************************')


