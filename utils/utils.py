import os
import dgl
import random
import torch as th
import pandas as pd
import math


def load_to_dgl_graph(dataset):
    edges = pd.read_csv(os.path.join('../data/', dataset, '{}.txt'.format(dataset)), sep=' ',
                        names=['start_idx', 'end_idx', 'time'])

    src_nid = edges.start_idx.to_numpy()
    dst_nid = edges.end_idx.to_numpy()

    graph = dgl.graph((src_nid, dst_nid))
    graph.edata['time'] = th.Tensor(edges.time.tolist())

    node_feat = position_encoding(max_len=graph.num_nodes(), emb_size=128)

    return graph, node_feat


def dataloader(dataset):
    edges = pd.read_csv(os.path.join('../data/', dataset, '{}.txt'.format(dataset)), sep=' ',
                        names=['start_idx', 'end_idx', 'time'])
    label = pd.read_csv(os.path.join('../data/', dataset, 'node2label.txt'), sep=' ', names=['nodeidx', 'label'])

    src_nid = edges.start_idx.to_numpy()
    dst_nid = edges.end_idx.to_numpy()

    graph = dgl.graph((src_nid, dst_nid))

    labels = th.full((graph.number_of_nodes(),), -1).cuda()

    nodeidx, lab = label.nodeidx.tolist(), label.label.tolist()

    for i in range(len(nodeidx)):
        labels[nodeidx[i]] = lab[i] - min(lab)

    train_mask = th.full((graph.number_of_nodes(),), False)
    val_mask = th.full((graph.number_of_nodes(),), False)
    test_mask = th.full((graph.number_of_nodes(),), False)

    random.seed(2024)
    train_mask_index, val_mask_index, test_mask_index = th.LongTensor([]), th.LongTensor([]), th.LongTensor([])
    for i in range(min(labels), max(labels) + 1):
        index = [j for j in label[label.label == i].nodeidx.tolist()]
        random.shuffle(index)
        train_mask_index = th.cat((train_mask_index, th.LongTensor(index[:int(len(index) / 10)])), 0)
        val_mask_index = th.cat((val_mask_index, th.LongTensor(index[int(len(index) / 10):int(len(index) / 5)])), 0)
        test_mask_index = th.cat((test_mask_index, th.LongTensor(index[int(len(index) / 5):])), 0)

    train_mask.index_fill_(0, train_mask_index, True).cuda()
    val_mask.index_fill_(0, val_mask_index, True).cuda()
    test_mask.index_fill_(0, test_mask_index, True).cuda()
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    n_classes = label.label.nunique()

    return labels, train_idx, val_idx, test_idx, n_classes


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


def load_subtensor(node_feats, input_nodes, device):  # 加载子张量至指定的设备
    batch_inputs = node_feats[input_nodes].to(device)
    return batch_inputs