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
from torch_scatter import scatter

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
    for node in range(num_nodes):
        neighbors = adjacency_dict.get(node, [])
        # print(f"Node {node} is connected to nodes {neighbors}")
    # file_name = "adjacency_dict.pkl"
    #
    # # 使用 pickle.dump() 将字典保存到文件
    # with open(file_name, 'wb') as file:
    #     pickle.dump(adjacency_dict, file)
    return adjacency_dict


def cal_info(result,label,edge_index,adjacency_matrix,pr):
    pos = 0
    neg = 0
    pos_num=0
    neg_num=0
    adjacency_matrix = adjacency_matrix.long()
    adjacency_dict = get_nodes(edge_index)
    for node in result:
        neighbors = adjacency_dict.get(node, [])
        sorted_neighbors = sorted(neighbors, key=lambda x: pr[x], reverse=True)
        print('node', node)
        # 优化循环
        print(neighbors)
        # for i in neighbors:
        #     print(i,pr[i],len(adjacency_dict.get(i, [])))
        print(sorted_neighbors[0])
        print(pr[sorted_neighbors[0]],pr[sorted_neighbors[-1]])
        for i in adjacency_dict.get(sorted_neighbors[0], []) :
            pos_num += len(adjacency_dict.get(sorted_neighbors[0], []))
            # pos_num += (len(adjacency_dict.get(sorted_neighbors[0], [])))
            pos += torch.sum(label[i] == label[node]).item()
            # pos += torch.sum(label[adjacency_dict.get(sorted_neighbors[0], [])] == label[node]).item()
            # n = np.where(adjacency_matrix[i] == 1)[0]
            # print(len(n))

        for i in adjacency_dict.get(sorted_neighbors[-1], []) :
            neg_num += len(adjacency_dict.get(sorted_neighbors[-1], []))
            # neg_num += (len(adjacency_dict.get(sorted_neighbors[-1], [])))
            neg += torch.sum(label[i] == label[node]).item()
            # neg += torch.sum(label[adjacency_dict.get(sorted_neighbors[-1], [])] == label[node]).item()
        print('pos',pos)
        print('neg',neg)
    return pos,neg,pos_num,neg_num

def nodes_with_two_or_more_neighbors(adjacency_matrix):
    n = len(adjacency_matrix)
    result = []

    for i in range(n):
        neighbors = np.where(adjacency_matrix[i] == 1)[0]  # 寻找邻居节点的索引
        if len(neighbors) >=9 and len(neighbors) <=12 :
            result.append(i)

    return result

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
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
    data = data.to(device)
    label = data.y.view(-1)

    adjacency_matrix = torch.zeros(data.num_nodes, data.num_nodes)
    for i, j in zip(*data.edge_index):
        adjacency_matrix[i, j] = 1
        adjacency_matrix[j, i] = 1
    result = nodes_with_two_or_more_neighbors(adjacency_matrix)
    print("Nodes with two or more neighbors:", result)
    pr = compute_pr(data.edge_index, 0.85, 10)
    pr = pr.tolist()
    print('pr', pr)
    pos, neg,pos_num,neg_num=cal_info(result,label,data.edge_index,adjacency_matrix,pr)
    print('pos',pos)
    print('neg',neg)
    print('pos_num',pos_num)
    print('neg_num',neg_num)







