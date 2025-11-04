import numpy as np
import scipy.sparse as sp
import torch
from torch_scatter import scatter
from torch_geometric.utils import degree

# 计算PageRank值
def compute_pr(adj_matrix, damp=0.85, k=10):
    # 将scipy稀疏矩阵转换为边索引格式
    coo = adj_matrix.tocoo()
    edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
    
    num_nodes = adj_matrix.shape[0]
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], dim=0, dim_size=num_nodes, reduce='sum')
        x = (1 - damp) * x + damp * agg_msg
    
    return x

# 主函数：生成优化的正负样本
def generate_enhanced_samples(pos_num=5, neg_num=5, alpha=0.7, beta=0.3):
    print("Loading and processing relation matrices...")
    p = 4019  # 节点数量
    
    # 加载并归一化元路径关系矩阵
    pap = sp.load_npz("./acm/pap.npz")
    pap = pap / (pap.sum(axis=-1).reshape(-1, 1) + 1e-8)  # 避免除零
    
    psp = sp.load_npz("./acm/psp.npz")
    psp = psp / (psp.sum(axis=-1).reshape(-1, 1) + 1e-8)  # 避免除零
    
    # 合并关系矩阵，但保持元路径的区分
    # alpha和beta是两种元路径的权重
    all_relations = (alpha * pap + beta * psp).A.astype("float32")
    all_connections = (all_relations > 0).sum(-1)
    print(f"Connection statistics - Max: {all_connections.max()}, Min: {all_connections.min()}, Mean: {all_connections.mean()}")
    
    # 分别计算两种元路径的PageRank值
    print("Computing PageRank values...")
    pr_pap = compute_pr(pap)
    pr_psp = compute_pr(psp)
    
    # 综合考虑两种元路径的PageRank值
    pr_values = (alpha * pr_pap + beta * pr_psp).numpy()
    
    # 生成正样本矩阵
    print("Generating positive samples...")
    pos = np.zeros((p, p))
    k = 0
    
    for i in range(len(all_relations)):
        # 获取当前节点的所有非零连接
        connected_nodes = all_relations[i].nonzero()[0]
        
        if len(connected_nodes) > pos_num:
            # 结合关系强度和PageRank值选择正样本
            # 计算综合得分：关系强度 * (1 + PageRank归一化值)
            scores = all_relations[i, connected_nodes] * (1 + pr_values[connected_nodes]/pr_values.max())
            top_indices = np.argsort(-scores)[:pos_num]
            selected_nodes = connected_nodes[top_indices]
            pos[i, selected_nodes] = 1
            k += 1
        else:
            # 如果连接节点少于pos_num，全部作为正样本
            pos[i, connected_nodes] = 1
    
    # 生成负样本矩阵 - 使用难负样本策略
    print("Generating negative samples...")
    neg = np.zeros((p, p))
    
    for i in range(len(all_relations)):
        # 获取当前节点的所有非连接节点
        non_connected = np.where(all_relations[i] == 0)[0]
        
        if len(non_connected) > 0:
            # 计算结构相似性 - 基于共同邻居
            # 为每个非连接节点计算与当前节点的共同邻居数量
            common_neighbors = np.zeros(len(non_connected))
            
            for j, node in enumerate(non_connected):
                # 当前节点的邻居
                neighbors_i = set(all_relations[i].nonzero()[0])
                # 候选负样本节点的邻居
                neighbors_j = set(all_relations[node].nonzero()[0])
                # 计算共同邻居数量
                common_neighbors[j] = len(neighbors_i.intersection(neighbors_j))
            
            # 结合结构相似性和PageRank值选择难负样本
            # 优先选择结构相似但未直接连接的节点，这些是更有挑战性的负样本
            neg_scores = 0.5 * common_neighbors/max(common_neighbors.max(), 1) + 0.5 * pr_values[non_connected]/pr_values.max()
            top_neg_indices = np.argsort(-neg_scores)[:neg_num]
            selected_neg_nodes = non_connected[top_neg_indices]
            neg[i, selected_neg_nodes] = 1
    
    # 保存为稀疏矩阵
    pos_sparse = sp.coo_matrix(pos)
    neg_sparse = sp.coo_matrix(neg)
    
    sp.save_npz("pos.npz", pos_sparse)
    sp.save_npz("neg.npz", neg_sparse)
    
    print(f"Completed! Generated {k} nodes with optimized positive samples")
    print(f"Positive samples saved to pos.npz")
    print(f"Negative samples saved to neg.npz")
    
    return pos_sparse, neg_sparse

# 执行优化的采样
if __name__ == "__main__":
    pos, neg = generate_enhanced_samples(pos_num=5, neg_num=5)
    
    # 输出统计信息
    print(f"Positive sample matrix shape: {pos.shape}")
    print(f"Positive sample density: {pos.nnz / (pos.shape[0] * pos.shape[1]):.6f}")
    print(f"Negative sample matrix shape: {neg.shape}")
    print(f"Negative sample density: {neg.nnz / (neg.shape[0] * neg.shape[1]):.6f}")