import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader
import numpy as np
import scipy
import torch
from torch_scatter import scatter

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

def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg
    # ascending_order = torch.argsort(x)
    # descending_order = ascending_order.flip(0)
    return x

def get_nodes(edge_index):
    num_nodes = edge_index.max().item() + 1
    # 初始化一个字典，用于存储每个节点的相邻节点
    adjacency_dict = {}
    # 遍历边的信息，构建相邻节点的映射
    for i in range(edge_index.size(1)):
        src_node = edge_index[0, i].item()
        dst_node = edge_index[1, i].item()
        if src_node not in adjacency_dict:
            adjacency_dict[src_node] = []
        adjacency_dict[src_node].append(dst_node)

    # 输出每个节点的相邻节点
    # for node in range(num_nodes):
    #     neighbors = adjacency_dict.get(node, [])
        # print(f"Node {node} is connected to nodes {neighbors}")
    # file_name = "adjacency_dict.pkl"
    #
    # # 使用 pickle.dump() 将字典保存到文件
    # with open(file_name, 'wb') as file:
    #     pickle.dump(adjacency_dict, file)
    return adjacency_dict

# 计算两个节点之间的共同邻居数量，用于评估节点对的相似性和重要性
def get_CoNei(sorted_node_indices,num_pairs,node,adj):
    if num_pairs >= len(sorted_node_indices):
        return sorted_node_indices
    common_neighbors = (adj[node] & adj[sorted_node_indices]).sum(dim=1)
    normalized_importance = (torch.arange(1, len(sorted_node_indices) + 1).float() - 1) / (len(sorted_node_indices) - 1)
    normalized_importance=normalized_importance.to(device)
    # 对得分进行归一化（在 [0, 1] 范围内）
    normalized_scores = (common_neighbors - common_neighbors.min()) / (common_neighbors.max() - common_neighbors.min())

    # 计算综合排名（加权平均）
    weighted_average_rank =  (1 - normalized_importance) + normalized_scores
    sorted_indices = torch.argsort(weighted_average_rank, descending=True)
    sorted_node_indices=torch.tensor(sorted_node_indices).to(device)
    return torch.index_select(sorted_node_indices, 0, sorted_indices)


def get_pairs(edge_index, num_pairs=2):
    # 构建邻接矩阵
    # 创建一个大小为 data.num_nodes x data.num_nodes 的全零矩阵作为邻接矩阵
    # zip(*data.edge_index)会将edge_index的两行解压成两个独立的序列
    # 例如edge_index = [[0,1,2], [1,2,0]] 经过zip后变成 (0,1), (1,2), (2,0)
    # 这样可以方便地用两个变量i,j同时获取边的起点和终点
    adj = torch.zeros(data.num_nodes, data.num_nodes).to(device)
    for i, j in zip(*data.edge_index):
        adj[i, j] = 1
        adj[j, i] = 1
    adj=adj.to(dtype=torch.int)
    # 计算节点的PageRank
    pr=compute_pr(edge_index,0.85,10)
    # 构建节点的相邻节点字典
    adjacency_dict=get_nodes(edge_index)
    # torch.arange(0, len(pr))生成一个从0到pr长度-1的连续张量序列
    # 从0开始是因为我们需要与节点索引对齐，节点编号通常从0开始
    num = torch.arange(0, len(pr))
    num = num = torch.arange(0, len(pr))
    weighted_nodes = list(zip(pr.tolist(), num.tolist()))
    pairs_dict = {}
    sorted_node_indices= {}
    for node in range(edge_index.max().item() + 1):
        neighbors = adjacency_dict.get(node, [])
        selected_nodes = [(weight, node) for weight, node in weighted_nodes if node in neighbors]
        sorted_nodes = sorted(selected_nodes, key=lambda x: x[0], reverse=True)
        sorted_node_indices[node] = [node for _, node in sorted_nodes]
        c=get_CoNei(sorted_node_indices[node],num_pairs,node,adj)
        # pairs_dict[node]=[sorted_node_indices[node] if len(sorted_node_indices[node])<=num_pairs else sorted_node_indices[node][0:num_pairs]]
        # pairs_dict[node]=[[sorted_node_indices[node][0:num_pairs] for i in range(len(sorted_node_indices[node]))] if len(sorted_node_indices[node])>0 else[node for i in range(len(sorted_node_indices[node]))]]
        pairs_dict[node]=[c[i] if i<len(sorted_node_indices[node]) else node for i in range(num_pairs)]
    return pairs_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
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

    path = osp.expanduser('./datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    data = data.to(device)
    get_nodes(data.edge_index)
    num=1
    pairs_dict = get_pairs(data.edge_index, num)
    file_name = args.dataset + "pairs_dict.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(pairs_dict, file)
    print(args.dataset)
    print(num)
    print("=== Pairs Over ===")


