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

def retrain(x,threshold,epoch_retrain):
    adj = torch.zeros(data.num_nodes, data.num_nodes)
    for i, j in zip(*data.edge_index):
        adj[i, j] = 1
        adj[j, i] = 1
    att=Att_P(x)
    top=Top_P(adj,threshold,epoch_retrain)
    return att,top

def Att_P(x):
    citations_per_article = torch.sum(x, axis=1)
    total_citations = torch.sum(citations_per_article)
    n = int(total_citations / x.shape[0]) + 1
    zero_tensor = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        # indices = torch.randint(0, x.shape[1], size=(n,))
        indices = random.sample(range(x.shape[1]), n)
        zero_tensor[i, indices] = 1
    zero_tensor = zero_tensor.astype(np.float32)
    zero_tensor = torch.from_numpy(zero_tensor)
    # zero_tensor.dtype=np.float32
    return zero_tensor.to('cuda:1')

class MyModule(nn.Module):
    def __init__(self,input_size, output_size):
        super(MyModule, self).__init__()
        self.learnable_matrix = nn.Parameter(torch.randn(input_size, input_size))
        nn.init.xavier_uniform_(self.learnable_matrix)
    def forward(self, input_data):
        output_data = torch.matmul(input_data, self.learnable_matrix)
        return output_data
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
    z4 = model(shuf_fts, data.edge_index)
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

def Top_P(adj,threshold,epoch_retrain):
    learnable_matrix_module = MyModule(adj.shape[0], adj.shape[0]).to(device )
    input_data = torch.randn(adj.shape[0], adj.shape[0])
    # binary = learnable_matrix_module(input_data).to(device)
    # binary=torch.sigmoid(binary)
    re_optimizer = torch.optim.Adam(
        learnable_matrix_module.parameters(),
        lr=0.00001,
        weight_decay=weight_decay
    )
    # binary= (binary > threshold).float()
    adj = torch.zeros(data.num_nodes, data.num_nodes).to(device)
    for i, j in zip(*data.edge_index):
        adj[i, j] = 1
        adj[j, i] = 1
    adjacency_matrix = torch.eye(adj.shape[0], device=device)
    for i in range(epoch_retrain):
        re_optimizer.zero_grad()
        binary = torch.sigmoid(learnable_matrix_module(adj))
        matrix1 = adjacency_matrix / adjacency_matrix.sum(axis=1, keepdims=True)
        matrix2 = binary / binary.sum(axis=1, keepdims=True)
        b_a=binary - adjacency_matrix
        F = torch.norm(b_a.view(-1), p=2)
        # F=torch.sum(F)
        # F = np.linalg.norm((binary - adjacency_matrix).detach(), 'fro')
        KL = kl_divergence_rowwise(matrix1, matrix2)
        KL=torch.sum(KL)
        loss=F-KL
        print(f'(T) | Epoch={i:03d}, loss={loss:.4f}')
        loss.backward(retain_graph=True)
        re_optimizer.step()
    return (binary > threshold).float()

def get_Neg(x):
    idx = np.random.permutation(x.shape[0])
    shuf_fts = x[idx, :]
    attr_per = Att_P(x)
    return shuf_fts, attr_per


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', type=str, default='CiteSeer')
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
    shuf_fts, top_per = retrain(data.x, 0.54,5)
    top_per = torch.nonzero(top_per, as_tuple=False)
    top_per = torch.stack([top_per[:, 0], top_per[:, 1]], dim=0)
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

