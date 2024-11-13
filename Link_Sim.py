import time
import pandas as pd
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import dgl
import random
import torch.nn as nn

from LREvaluator_base import LREvaluator_base
from utils import get_edges_split, compute_bunch_matrices, compute_hodge_matrix
torch.autograd.set_detect_anomaly(True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
setup_seed(42)
def get_total_edges(data, type):
    if type == 'train':
        edges_pos = data.total_edges[:data.train_pos]
        index = np.random.randint(0, data.train_neg, data.train_pos)
        edges_neg = data.total_edges[data.train_pos:data.train_pos + data.train_neg][index]
        total_edges = np.concatenate((edges_pos, edges_neg))
        edges_y = torch.cat((data.total_edges_y[:data.train_pos],
                             data.total_edges_y[data.train_pos:data.train_pos + data.train_neg][index]))
        return total_edges, edges_y
    elif type == 'val':
        total_edges = data.total_edges[
                      data.train_pos + data.train_neg:data.train_pos + data.train_neg + data.val_pos + data.val_neg]
        edges_y = data.total_edges_y[
                  data.train_pos + data.train_neg:data.train_pos + data.train_neg + data.val_pos + data.val_neg]
        return total_edges, edges_y
    elif type == 'test':
        total_edges = data.total_edges[
                      data.train_pos + data.train_neg + data.val_pos + data.val_neg:]
        edges_y = data.total_edges_y[
                  data.train_pos + data.train_neg + data.val_pos + data.val_neg:]
        return total_edges, edges_y
    return 'error'
def split_train_valid_test(data):
    val_prop = 0.1
    test_prop = 0.8
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_edges_split(data, val_prop = val_prop, test_prop = test_prop)
    total_edges = np.concatenate((train_edges,train_edges_false,val_edges,val_edges_false,test_edges,test_edges_false))
    data.train_pos,data.train_neg = len(train_edges),len(train_edges_false)
    data.val_pos, data.val_neg = len(val_edges), len(val_edges_false)
    data.test_pos, data.test_neg = len(test_edges), len(test_edges_false)
    data.total_edges = total_edges
    data.total_edges_y = torch.cat((torch.ones(len(train_edges)), torch.zeros(len(train_edges_false)), torch.ones(len(val_edges)), torch.zeros(len(val_edges_false)),torch.ones(len(test_edges)), torch.zeros(len(test_edges_false)))).long()
    edge_list = np.array(data.edge_index).T.tolist()
    for edges in val_edges:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            # edge_list.remove([edges[1], edges[0]])
    for edges in test_edges:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            # edge_list.remove([edges[1], edges[0]])
    data.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)
    return data

def sample_edge_index(data):
    random_edge_num = 1000
    indices = np.random.choice((data.edge_index).size(1), (random_edge_num,), replace=False)
    indices = np.sort(indices)
    sample_data_edge_index = data.edge_index[:, indices]
    # sample_data_edge_index = data.edge_index
    boundary_matrix0_, boundary_matrix1_ = compute_hodge_matrix(data, sample_data_edge_index)
    L0u, L1f = compute_bunch_matrices(boundary_matrix0_, boundary_matrix1_)
    L0u = torch.tensor(L0u, dtype=torch.float32)
    L1f = torch.tensor(L1f, dtype=torch.float32)
    return L0u, L1f

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers, size):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))
        self.weights_L_0 = nn.Parameter(torch.FloatTensor(int(size), hidden_dim))
        self.weights_L_1 = nn.Parameter(torch.FloatTensor(int(size), hidden_dim))
        self.weights_off_diagonal = nn.Parameter(torch.FloatTensor(int(size), int(size)))
        self.weights_sim = nn.Parameter(torch.FloatTensor(int(size*2), hidden_dim))
        self.embeddings_sim = nn.Parameter(torch.FloatTensor(input_dim, int(size*2)))
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear_1 = torch.nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
    def Block_Hodge(self, z, L0, L1):
        L0_r = torch.matrix_power(L0, 2)
        L1_r = torch.matrix_power(L1, 2)
        diag_zeros = torch.zeros(L0.size()).to(L0.device)
        upper_block = torch.cat([L0_r, diag_zeros], dim=1)
        lower_block = torch.cat([diag_zeros, L1_r], dim=1)
        sim_block = torch.cat([upper_block, lower_block], dim=0)
        embeddings_sim = torch.matmul(z, self.embeddings_sim)
        s_emb_sim_ = torch.matmul(embeddings_sim, sim_block)
        s_emb_sim = torch.matmul(s_emb_sim_, self.weights_sim)
        s_emb_sim = s_emb_sim.renorm_(2, 0, 1)
        return s_emb_sim

    def forward(self, x, edge_index, org_edge_index, L0, L1, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        g_emb = z
        s_emb = self.Block_Hodge(x, L0, L1)
        return [g_emb, s_emb]
def compute_hodge_laplacian(L0, L1):
    L0_Laplacian = torch.mm(L0, L0.t())
    L1_Laplacian = torch.mm(L1.t(), L1)
    zeros_top = torch.zeros(L0_Laplacian.size(0), L1_Laplacian.size(1), device=L0.device)
    zeros_bottom = torch.zeros(L1_Laplacian.size(0), L0_Laplacian.size(1), device=L0.device)
    top = torch.cat((L0_Laplacian, zeros_top), dim=1)
    bottom = torch.cat((zeros_bottom, L1_Laplacian), dim=1)
    H = torch.cat((top, bottom), dim=0)
    return H
def random_mask(L, drop_prob):
    drop_mask = torch.empty(
        (L.size(1), ),
        dtype=torch.float32,
        device=L.device).uniform_(0, 1) < drop_prob
    L = L.clone()
    L[:, drop_mask] = 0
    return L

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, L0, L1, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        org_edge_index = edge_index
        L0_1 = random_mask(L0, 0.1)
        L0_2 = random_mask(L0, 0.2)
        L1_1 = random_mask(L1, 0.2)
        L1_2 = random_mask(L1, 0.1)
        emb = self.encoder(x, edge_index, org_edge_index, L0, L1, edge_weight)
        emb1 = self.encoder(x1, edge_index1, org_edge_index, L0_1, L1_1, edge_weight1)
        emb2 = self.encoder(x2, edge_index2, org_edge_index, L0_2, L1_2, edge_weight2)
        return emb, emb1, emb2, L0_1, L0_2, L1_1, L1_2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def laplacian_contrastive_loss(L0_1, L1_1, L0_2, L1_2, alpha=0.5, beta=0.5):
    L0_1, L1_1, L0_2, L1_2 = L0_1.to(torch.float32), L1_1.to(torch.float32), L0_2.to(torch.float32), L1_2.to(
        torch.float32)

    eigvals_L0_1, eigvecs_L0_1 = torch.symeig(L0_1, eigenvectors=True)
    eigvals_L1_1, eigvecs_L1_1 = torch.symeig(L1_1, eigenvectors=True)
    eigvals_L0_2, eigvecs_L0_2 = torch.symeig(L0_2, eigenvectors=True)
    eigvals_L1_2, eigvecs_L1_2 = torch.symeig(L1_2, eigenvectors=True)

    lambda_loss = torch.mean((eigvals_L0_1 - eigvals_L0_2) ** 2) + torch.mean((eigvals_L1_1 - eigvals_L1_2) ** 2)

    vec_similarity_L0 = torch.sum(eigvecs_L0_1 * eigvecs_L0_2) / (torch.norm(eigvecs_L0_1) * torch.norm(eigvecs_L0_2))
    vec_similarity_L1 = torch.sum(eigvecs_L1_1 * eigvecs_L1_2) / (torch.norm(eigvecs_L1_1) * torch.norm(eigvecs_L1_2))
    vec_loss = 2 - vec_similarity_L0 - vec_similarity_L1

    total_loss = alpha * lambda_loss + beta * vec_loss
    return total_loss


def regularized_eigh(matrix, reg_lambda=1e-7):
    regularization = reg_lambda * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    regularized_matrix = matrix + regularization
    eigvals, eigvecs = torch.linalg.eigh(regularized_matrix)
    return eigvals, eigvecs

def improved_laplacian_contrastive_loss(L0_1, L1_1, L0_2, L1_2, alpha=0.5, beta=0.5, gamma=0.5):
    L0_1, L1_1, L0_2, L1_2 = L0_1.float(), L1_1.float(), L0_2.float(), L1_2.float()
    eigvals_L0_1, eigvecs_L0_1 = regularized_eigh(L0_1)
    eigvals_L1_1, eigvecs_L1_1 = regularized_eigh(L1_1)
    eigvals_L0_2, eigvecs_L0_2 = regularized_eigh(L0_2)
    eigvals_L1_2, eigvecs_L1_2 = regularized_eigh(L1_2)
    lambda_loss = torch.mean((eigvals_L0_1 / eigvals_L0_1.max() - eigvals_L0_2 / eigvals_L0_2.max()) ** 2) + \
                  torch.mean((eigvals_L1_1 / eigvals_L1_1.max() - eigvals_L1_2 / eigvals_L1_2.max()) ** 2)
    vec_similarity_L0 = (torch.sum(eigvecs_L0_1 * eigvecs_L0_2) /
                         (torch.norm(eigvecs_L0_1) * torch.norm(eigvecs_L0_2)))
    vec_similarity_L1 = (torch.sum(eigvecs_L1_1 * eigvecs_L1_2) /
                         (torch.norm(eigvecs_L1_1) * torch.norm(eigvecs_L1_2)))
    vec_loss = 2 - vec_similarity_L0 - vec_similarity_L1
    alignment_loss = torch.mean(torch.abs(eigvecs_L0_1 - eigvecs_L0_2)) + \
                     torch.mean(torch.abs(eigvecs_L1_1 - eigvecs_L1_2))
    total_loss = alpha * lambda_loss + beta * vec_loss + gamma * alignment_loss
    return total_loss

def train(encoder_model, contrast_model, data, optimizer, L0, L1):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2, L0_1, L0_2, L1_1, L1_2 = encoder_model(data.x, data.edge_index, L0, L1, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1[0], z2[0]]]
    h3, h4 = [encoder_model.project(x) for x in [z1[1], z2[1]]]
    # loss = contrast_model(h1, h2) * 0.5 + contrast_model(h3, h4) * 0.5 + laplacian_contrastive_loss(L0_1, L0_2, L1_1, L1_2) * 0.5
    # h1, h2 = [encoder_model.project(x) for x in [z1[0], z2[0]]]
    # h3, h4 = [encoder_model.project(x) for x in [z1[1], z2[1]]]
    # l1 = contrast_model(h1, h2) * 0.5
    # l2 = contrast_model(h3, h4) * 0.5
    # l3 = laplacian_contrastive_loss(L0_1, L0_2, L1_1, L1_2) * 0.5
    loss = contrast_model(h1, h2) * 0.5 + contrast_model(h3, h4) * 0.5 + laplacian_contrastive_loss(L0_1, L0_2, L1_1, L1_2) * 0.5
    # print('l1:{}, l2:{}, l3:{}'.format(l1, l2, l3))
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data, L0, L1):
    encoder_model.eval()
    z, _, _, _, _, _, _ = encoder_model(data.x, data.edge_index, L0, L1, data.edge_attr)
    data = split_train_valid_test(data.cpu())
    split = get_split(num_samples=z[0].size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator_base()(z, data, split)
    return result


def main():
    device = torch.device('cuda')
    # dataset = Planetoid(root='dataset', name='Cora', transform=T.NormalizeFeatures())
    # data = dataset[0]
    # print(data.x.shape)
    # data = torch.load('dataset/Metting/Metting.pt')
    # data = torch.load('dataset/Phonecalls/Phonecalls.pt')
    data = torch.load('dataset/Disease/Disease.pt')
    L0, L1 = sample_edge_index(data)
    L0 = L0.to(device)
    L1 = L1.to(device)
    data = data.to(device)
    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    gconv = GConv(input_dim=data.x.shape[1], hidden_dim=64, activation=torch.nn.ReLU, num_layers=2, size=L0.shape[0]).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=64, proj_dim=64).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.6), mode='L2L', intraview_negs=True).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.01, weight_decay=0)
    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, data, optimizer, L0, L1)
            pbar.set_postfix({'loss': loss})
            pbar.update()
    test_result = test(encoder_model, data, L0, L1)
    print(f'(E): Best test_roc={test_result["best_test_roc"]:.4f}, test_acc={test_result["best_test_acc"]:.4f}')
    return test_result["best_test_roc"], test_result["best_test_acc"]
def simple_progress_bar(total_steps):
    for i in range(total_steps):
        percent_complete = (i + 1) / total_steps * 100
        progress = int(percent_complete // 2)
        print(f'\rProgress: [{"#" * progress}{" " * (50 - progress)}] {percent_complete:.2f}%', end='', flush=True)
        time.sleep(0.1)
    print('\n')
if __name__ == '__main__':
    main()
    # pe1s = [0.1, 0.3, 0.5, 0.7, 0.9]
    # pe2s = [0.1, 0.3, 0.5, 0.7, 0.9]
    # pf1s = [0.1, 0.3, 0.5, 0.7, 0.9]
    # pf2s = [0.1, 0.3, 0.5, 0.7, 0.9]
    # total_iterations = len(pe1s) * len(pe2s) * len(pf1s) * len(pf2s)
    # current_iteration = 0
    # results = []
    # for pe1 in pe1s:
    #     for pe2 in pe2s:
    #         for pf1 in pf1s:
    #             for pf2 in pf2s:
    #                 torch.cuda.empty_cache()
    #                 roc, acc = main(pe1, pe2, pf1, pf2)
    #                 results.append(
    #                     {'pe1': pe1, 'pe2': pe2, 'pf1': pf1, 'pf2': pf2,
    #                      'test_roc': roc,
    #                      'test_acc': acc})
    #                 current_iteration += 1
    #                 percent_complete = (current_iteration / total_iterations) * 100
    #                 print(f'Current progress: {percent_complete:.2f}%')
    # df = pd.DataFrame(results)
    # df.to_excel('Airports_result.xlsx', index=False)






