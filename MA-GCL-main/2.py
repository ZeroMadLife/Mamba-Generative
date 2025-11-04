import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.init as init
from torch_scatter import scatter_add
import torch.nn as nn
from torch_geometric.utils import dropout_adj, degree, to_undirected, get_laplacian
import torch.nn.functional as F
import networkx as nx
from scipy.sparse.linalg import eigs, eigsh

from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE, NewGConv, NewEncoder, NewGRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from utils import normalize_adjacency_matrix, create_adjacency_matrix, load_adj_neg, Rank
import pickle


def retrain(x,epoch,eps=1e-10):
    model=MyModule(x.shape[0],x.shape[1])
    re_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=weight_decay
    )
    for i in range(epoch):
        re_optimizer.zero_grad()
        matrix=model().to(device)
        if torch.isnan(matrix).any().item():
            print("Check and clean your data.")
        x[:,0] +=eps
        matrix1 = x / (x.sum)(axis=1, keepdims=True)
        if torch.isnan(matrix1).any().item():
            print("Check and clean your data.")
        matrix2 = matrix / matrix.sum(axis=1, keepdims=True)
        if torch.isnan(matrix2).any().item():
            print("Check and clean your data.")
        kl_divergence=kl_divergence_rowwise(matrix1,matrix2)
        if torch.isnan(kl_divergence).any().item():
            print("Check and clean your data.")
        KL=kl_divergence
        b_a = x - matrix
        F = torch.norm(b_a.view(-1), p=2)-torch.norm(matrix.view(-1), p=2)
        KL = torch.sum(kl_divergence)
        loss = F 
        loss.backward()
        re_optimizer.step()
        print(f'(T) | Epoch={i:03d}, loss={loss:.4f}')
    return matrix.detach()
class MyModule(nn.Module):
    def __init__(self,n, h):
        super(MyModule, self).__init__()
        self.learnable_matrix = nn.Parameter(torch.randn(n, h))
        nn.init.uniform_(self.learnable_matrix, a=-0.1, b=0.1)
    def forward(self):
        if torch.isnan(self.learnable_matrix).any().item() or torch.isnan(self.learnable_matrix).any().item() or torch.isinf(
                self.learnable_matrix).any().item() or torch.isinf(self.learnable_matrix).any().item():
            print("Check and clean your data.")
        return F.relu(self.learnable_matrix)
def train():
    model.train()
    # view_learner.eval()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]  # adjacency with edge droprate 2
    x_1 = drop_feature(data.x, drop_feature_rate_1)  # 3
    x_2 = drop_feature(data.x, drop_feature_rate_2)  # 4

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    z3 = model(shuf_fts, data.edge_index)
    z4 = model(att_p,data.edge_index)

    loss = model.loss(z1, z2, z3, z4, tensor, batch_size=None)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        accs_1 = []
        accs_2 = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or args.dataset == 'PubMed':
            acc = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=split)['acc']
            # acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=0)['acc']
        else:
            acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=0)['acc']
        # acc_2 = log_regression(z2, dataset, evaluator2, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    return acc  # , acc_1, acc_2

def kl_divergence_rowwise(matrix1, matrix2,eps=1e-10):
    # 遍历每一行，计算KL散度
    kl_divergences = torch.zeros(matrix1.shape[0]).to(device)
    for i in range(matrix1.shape[0]):
        p_i = matrix1[i, :]
        q_i = matrix2[i, :]
        kl_divergences[i] = torch.sum(p_i * torch.log((p_i+ eps) / (q_i+ eps)))
    return kl_divergences

def get_Neg(x,epoch):
    idx = np.random.permutation(x.shape[0])
    shuf_fts = x[idx, :]
    attr_per = retrain(x,epoch)
    return shuf_fts, attr_per


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--config', type=str, default='param.yaml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=str, default='train,eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(args.seed)
    use_nni = args.config == 'nni'
    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = config['activation']
    base_model = config['base_model']
    num_layers = config['num_layers']
    dataset = args.dataset
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    drop_scheme = config['drop_scheme']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    rand_layers = config['rand_layers']

    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    data = data.to(device)

    adj = 0
    shuf_fts, att_p = get_Neg(data.x,50)
    print('Retrain Finish')
    if args.dataset == 'Cora' or args.dataset == 'CiteSeer' or args.dataset == 'PubMed': split = (
    data.train_mask, data.val_mask, data.test_mask)

    encoder = Encoder(dataset.num_features, num_hidden, get_activation(activation),
                      base_model=GCNConv, k=num_layers).to(device)

    model = GRACE(encoder, num_hidden, num_proj_hidden, tau).to(device)
    file_name = args.dataset + "pairs_dict.pkl"
    with open(file_name, 'rb') as file:
        pair_dict = pickle.load(file)
    values_list = list(pair_dict.values())
    tensor = torch.tensor(values_list).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    for epoch in range(1, num_epochs + 1):
        loss = train()
        log = args.verbose.split(',')
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
        if epoch % 100 == 0:
            acc = test()

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {acc}')
            d = str(str(epoch) + ' ' + str(acc))
            with open('output.txt', mode='a') as filename:
                filename.write(d)
                filename.write('\n')

    acc = test(final=True)

    if 'final' in log:
        print(f'{acc}')

