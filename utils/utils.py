import os
import dgl
import random
import torch as th
import math


def load_dataset(dataset, num_nodes, emb_size=128):
    hetero_graph_list = dgl.load_graphs(os.path.join('../data/', dataset, '{}.bin'.format(dataset)))[0]  # 异质图列表
    node_feat = position_encoding(max_len=num_nodes, emb_size=emb_size)  # 节点特征
    return hetero_graph_list, node_feat


def split_dataset(num_nodes_list):  # 划分数据集
    num_nodes = sum(num_nodes_list)  # 节点数
    train_mask = th.full((num_nodes,), False)  # 训练集
    val_mask = th.full((num_nodes,), False)  # 验证集
    test_mask = th.full((num_nodes,), False)  # 测试集

    random.seed(2024)
    train_mask_index, val_mask_index, test_mask_index = th.LongTensor([]), th.LongTensor([]), th.LongTensor([])
    for i in range(len(num_nodes_list)):
        start_index = sum(num_nodes_list[0:i])  # 起始 index
        ids = range(start_index, start_index + num_nodes_list[i])  # 待划分 id
        random.shuffle(ids)  # 打乱
        # 训练集取 60%
        train_mask_index = th.cat((train_mask_index, th.LongTensor(ids[:int(len(ids) * 0.6)])), 0)
        # 验证集取 20%
        val_mask_index = th.cat((val_mask_index, th.LongTensor(ids[int(len(ids) * 0.6):int(len(ids) * 0.8)])), 0)
        # 测试集取 20%
        test_mask_index = th.cat((test_mask_index, th.LongTensor(ids[int(len(ids) * 0.8):])), 0)

    train_mask.index_fill_(0, train_mask_index, True).cuda()  # 训练集
    val_mask.index_fill_(0, val_mask_index, True).cuda()  # 验证集
    test_mask.index_fill_(0, test_mask_index, True).cuda()  # 测试集

    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()  # 训练集
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()  # 验证集
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()  # 测试集

    return train_idx, val_idx, test_idx


def position_encoding(max_len, emb_size):  # 位置编码
    pe = th.zeros(max_len, emb_size)
    position = th.arange(0, max_len).unsqueeze(1)

    div_term = th.exp(th.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))

    pe[:, 0::2] = th.sin(position * div_term)
    pe[:, 1::2] = th.cos(position * div_term)
    return pe


def sampling_layer(snapshots, views, strategy='random'):  # 采样层
    samples = []  # 采样结果
    random.seed(2024)

    if strategy == 'random':  # 随机采样
        samples = random.sample(range(0, snapshots), views)  # 随机采取 views 个样本
    elif strategy == 'sequential':  # 顺序采样
        samples = random.sample(range(0, snapshots - views + 1), 1)  # 随机采取 1 个样本
        start = samples[0]
        for i in range(1, views):
            samples.append(start + i)  # 按顺序取剩下的样本

    return samples
