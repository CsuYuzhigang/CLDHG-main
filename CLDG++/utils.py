import os
import dgl
import random
import torch as th
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import euclidean

random.seed(24)


def load_to_dgl_graph(dataset, s):  # 处理图数据并进行异常注入实验
    edges = pd.read_csv(os.path.join('../Data/', dataset, '{}.txt'.format(dataset)), sep=' ',
                        names=['start_idx', 'end_idx', 'time'])  # 边

    src_nid = edges.start_idx.to_numpy()  # 源节点
    dst_nid = edges.end_idx.to_numpy()  # 目标节点

    graph = dgl.graph((src_nid, dst_nid))  # 构建图
    graph.edata['time'] = th.Tensor(edges.time.tolist())  # 存储边中的时间信息

    node_feat = position_encoding(max_len=graph.num_nodes(), emb_size=128)  # 节点位置编码

    # m: num of fully connected nodes
    # n: num of fully connected clusters
    # k: another 𝑘 nodes as a candidate set
    m, n, k = 15, 20, 50

    if dataset == 'bitcoinotc' or dataset == 'bitotc' or dataset == 'bitalpha':
        n = 10
    elif dataset == 'dblp' or dataset == 'tax':
        n = 20
    elif dataset == 'tax51' or dataset == 'reddit':
        n = 200
    inject_graph, inject_node_feat, anomaly_label = inject_anomaly(graph, node_feat, m, n, k, s)  # 异常注入

    return inject_graph, inject_node_feat, anomaly_label


def inject_anomaly(g, feat, m, n, k, s):  # 异常注入
    num_node = g.num_nodes()  # 图节点数
    all_idx = list(range(g.num_nodes()))  # 索引列表
    random.shuffle(all_idx)  # 打乱索引列表
    anomaly_idx = all_idx[:m * n * 2]  # 前 m * n * 2 个为异常索引

    structure_anomaly_idx = anomaly_idx[:m * n]  # 结构异常
    attribute_anomaly_idx = anomaly_idx[m * n:]  # 属性异常
    label = np.zeros((num_node, 1), dtype=np.uint8)
    label[anomaly_idx, 0] = 1  # 异常节点全 1, 其余节点全 0

    str_anomaly_label = np.zeros((num_node, 1), dtype=np.uint8)  # 标记结构异常节点
    str_anomaly_label[structure_anomaly_idx, 0] = 1
    attr_anomaly_label = np.zeros((num_node, 1), dtype=np.uint8)  # 标记属性异常节点
    attr_anomaly_label[attribute_anomaly_idx, 0] = 1

    # Disturb structure
    print('Constructing structured anomaly nodes...')  # 构造结构异常节点
    u_list, v_list, t_list = [], [], []  # 存储新边的起始节点、终止节点和时间戳
    max_time, min_time = max(g.edata['time'].tolist()), min(g.edata['time'].tolist())
    for n_ in range(n):  # 在每个完全连接的簇中，为节点对添加边
        current_nodes = structure_anomaly_idx[n_ * m:(n_ + 1) * m]
        t = random.uniform(min_time, max_time)
        for i in current_nodes:
            for j in current_nodes:
                u_list.append(i)
                v_list.append(j)
                t_list.append(t)

    ori_num_edge = g.num_edges()  # 原始边数
    g = dgl.add_edges(g, th.tensor(u_list), th.tensor(v_list), {'time': th.tensor(t_list)})  # 添加新边

    num_add_edge = g.num_edges() - ori_num_edge  # 添加的边数
    print('Done. {:d} structured nodes are constructed. ({:.0f} edges are added) \n'.format(len(structure_anomaly_idx),
                                                                                            num_add_edge))

    # Disturb attribute
    print('Constructing attributed anomaly nodes...')  # 构造属性异常节点
    feat_list = []
    ori_feat = feat
    attribute_anomaly_idx_list = split_list(attribute_anomaly_idx, s)  # 打乱属性异常节点的特征, 将属性异常节点索引拆分成 s 个子集
    for lst in attribute_anomaly_idx_list:  # 对每个属性异常节点子集, 选择 k 个候选节点, 计算当前节点特征与候选节点特征之间的欧几里得距离, 将最大距离的候选节点特征赋给当前节点
        feat = ori_feat
        for i_ in lst:
            picked_list = random.sample(all_idx, k)
            max_dist = 0
            for j_ in picked_list:
                cur_dist = euclidean(ori_feat[i_], ori_feat[j_])
                if cur_dist > max_dist:
                    max_dist = cur_dist
                    max_idx = j_
            feat[i_] = feat[max_idx]
        feat_list.append(feat)
    print('Done. {:d} attributed nodes are constructed. \n'.format(len(attribute_anomaly_idx)))

    return g, feat_list, label


def dataloader(dataset):  # 加载数据
    edges = pd.read_csv(os.path.join('../Data/', dataset, '{}.txt'.format(dataset)), sep=' ',
                        names=['start_idx', 'end_idx', 'time'])  # 边
    label = pd.read_csv(os.path.join('../Data/', dataset, 'node2label.txt'), sep=' ', names=['nodeidx', 'label'])  # 标签

    src_nid = edges.start_idx.to_numpy()  # 源节点
    dst_nid = edges.end_idx.to_numpy()  # 目标节点

    graph = dgl.graph((src_nid, dst_nid))  # 构建图

    labels = th.full((graph.number_of_nodes(),), -1).cuda()  # 存储标签

    nodeidx, lab = label.nodeidx.tolist(), label.label.tolist()  # 索引列表, 标签列表

    for i in range(len(nodeidx)):
        labels[nodeidx[i]] = lab[i] - min(lab)

    train_mask = th.full((graph.number_of_nodes(),), False)  # 训练集掩码
    val_mask = th.full((graph.number_of_nodes(),), False)  # 验证集掩码
    test_mask = th.full((graph.number_of_nodes(),), False)  # 测试集掩码

    random.seed(24)
    train_mask_index, val_mask_index, test_mask_index = th.LongTensor([]), th.LongTensor([]), th.LongTensor([])
    for i in range(min(labels), max(labels) + 1):  # 划分数据集
        index = [j for j in label[label.label == i].nodeidx.tolist()]
        random.shuffle(index)
        train_mask_index = th.cat((train_mask_index, th.LongTensor(index[:int(len(index) / 10)])), 0)
        val_mask_index = th.cat((val_mask_index, th.LongTensor(index[int(len(index) / 10):int(len(index) / 5)])), 0)
        test_mask_index = th.cat((test_mask_index, th.LongTensor(index[int(len(index) / 5):])), 0)

    train_mask.index_fill_(0, train_mask_index, True).cuda()
    val_mask.index_fill_(0, val_mask_index, True).cuda()
    test_mask.index_fill_(0, test_mask_index, True).cuda()
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()  # 训练集索引
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()  # 验证集索引
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()  # 测试集索引
    n_classes = label.label.nunique()  # 类别数

    return labels, train_idx, val_idx, test_idx, n_classes


def position_encoding(max_len, emb_size):  # 位置编码
    pe = th.zeros(max_len, emb_size)  # 存储位置编码
    position = th.arange(0, max_len).unsqueeze(1)  # 位置索引

    div_term = th.exp(th.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))  # 存有位置编码的缩放因子

    pe[:, 0::2] = th.sin(position * div_term)  # 对偶数列赋值
    pe[:, 1::2] = th.cos(position * div_term)  # 对奇数列赋值
    return pe


def split_list(lst, s):
    avg_length = len(lst) // s
    remainder = len(lst) % s
    result = [lst[i * avg_length + min(i, remainder):(i + 1) * avg_length + min(i + 1, remainder)] for i in range(s)]
    return result


def sampling_layer(snapshots, views, span, strategy):  # 采样
    T = []
    if strategy == 'random':  # 随机策略
        T = [random.uniform(0, span * (snapshots - 1) / snapshots) for _ in range(views)]
    elif strategy == 'low_overlap':  # 低重叠策略
        if (0.75 * views + 0.25) > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        start = random.uniform(0, span - (0.75 * views + 0.25) * span / snapshots)
        T = [start + (0.75 * i * span) / snapshots for i in range(views)]
    elif strategy == 'high_overlap':  # 高重叠策略
        if (0.25 * views + 0.75) > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        start = random.uniform(0, span - (0.25 * views + 0.75) * span / snapshots)
        T = [start + (0.25 * i * span) / snapshots for i in range(views)]
    elif strategy == 'sequential':  # 顺序策略
        T = [span * i / snapshots for i in range(snapshots)]
        ori_T = T
        if views > snapshots:
            return "The number of sampled views exceeds the maximum value of the current policy."
        T = random.sample(T, views)
        T_idx = [ori_T.index(i) for i in T]

    return T, T_idx
