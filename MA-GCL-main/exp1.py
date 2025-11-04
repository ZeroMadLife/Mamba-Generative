import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import torch
from torch_geometric.nn import GCNConv, GATConv
from scipy.sparse import coo_matrix

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




def get_Nei(adj,label):
    res = [0] * 8
    num = [0] * 8
    for i in range(adj.shape[0]):
        _res = 0.0
        _num=0
        for j in range(adj.shape[0]):
            _num += adj[i][j]
            if label[i] == label[j] and adj[i][j] == 1:
                _res += 1
        res[label[i]] = res[label[i]] + _res
        num[label[i]] = num[label[i]] + _num
        print('node:'+str(i))
    for i in range(8):
        print(num[i])
    return res,num










if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Amazon-Photo')
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
    label = data.y.view(-1)
    # adj = coo_matrix(
    #     (np.ones(data.num_edges), (data.edge_index[0].cpu().numpy(), data.edge_index[1].cpu().numpy())),
    #     shape=(data.num_nodes, data.num_nodes))
    adjacency_matrix = torch.zeros(data.num_nodes, data.num_nodes)
    for i, j in zip(*data.edge_index):
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    res, num = get_Nei(adjacency_matrix, label)
    print(res)
    for i in range(8):
        print(res[i] / num[i])







