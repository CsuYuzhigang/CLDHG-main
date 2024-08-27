import torch.nn as thnn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv, HeteroLinear


class LogReg(thnn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = thnn.Linear(hid_dim, out_dim)  # 线性层

    def forward(self, x):
        ret = self.fc(x)  # 前向传播
        return ret


class MLPLinear(thnn.Module):  # 线性层
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = HeteroLinear(in_dim, out_dim)  # 线性层 1
        self.linear2 = HeteroLinear(out_dim, out_dim)  # 线性层 2
        self.act = thnn.LeakyReLU(0.2)  # LeakyReLU 激活函数
        self.reset_parameters()  # 初始化参数

    def reset_parameters(self):  # 初始化参数
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):  # 前向传播
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x


class GraphConvModel(thnn.Module):

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 n_layers,
                 n_classes,
                 norm,
                 activation,
                 readout,
                 dropout):
        super(GraphConvModel, self).__init__()
        self.in_feats = in_feats
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.norm = norm
        self.activation = activation
        self.readout = readout
        self.dropout = thnn.Dropout(dropout)

        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(GraphConv(in_feats=self.in_feats,
                                     out_feats=self.hidden_dim,
                                     norm=self.norm,
                                     activation=self.activation,
                                     allow_zero_in_degree=True))  # 第一个图卷积层
        for i in range(1, (self.n_layers - 1)):
            self.layers.append(GraphConv(in_feats=self.hidden_dim,
                                         out_feats=self.hidden_dim,
                                         norm=self.norm,
                                         activation=self.activation,
                                         allow_zero_in_degree=True))  # 中间的图卷积层
        self.layers.append(GraphConv(in_feats=self.hidden_dim,
                                     out_feats=self.n_classes,
                                     norm=self.norm,
                                     activation=self.activation))  # 最后一个图卷积层
        self.linear = thnn.Linear(self.n_classes, self.n_classes)  # 添加一个线性层，用于将最后的特征进行线性变换

        self.act = thnn.LeakyReLU(0.2)  # LeakyReLU 激活函数

    def forward(self, blocks, features):  # 前向传播
        h = features
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.dropout(h)

        h = self.act(F.normalize(self.linear(h), p=2, dim=1))

        return h


class HeteroGraphConvModel(thnn.Module):
    def __init__(self,
                 edge_types,
                 in_feats,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 norm,
                 activation,
                 aggregate='sum',
                 readout='max',
                 dropout=0):
        super(HeteroGraphConvModel, self).__init__()
        self.edge_types = edge_types
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.norm = norm
        self.activation = activation
        self.aggregate = aggregate
        self.readout = readout
        self.dropout = thnn.Dropout(dropout)
        self.layers = thnn.ModuleList()

        # build multiple layers
        self.layers.append(HeteroGraphConv(
            mods={edge: GraphConv(self.in_feats, self.hidden_dim, norm=self.norm,
                                  activation=self.activation, allow_zero_in_degree=True) for edge in self.edge_types},
            aggregate=self.aggregate))  # 第一个异质图卷积层
        for i in range(1, (self.n_layers - 1)):
            self.layers.append(HeteroGraphConv(
                mods={edge: GraphConv(self.hidden_dim, self.hidden_dim, norm=self.norm,
                                      activation=self.activation, allow_zero_in_degree=True) for edge in self.edge_types},
                aggregate=self.aggregate))  # 中间的异质图卷积层
        self.layers.append(HeteroGraphConv(
            mods={edge: GraphConv(self.hidden_dim, self.output_dim, norm=self.norm,
                                  activation=self.activation, allow_zero_in_degree=True) for edge in self.edge_types},
            aggregate=self.aggregate))  # 最后一个异质图卷积层

        self.linear = HeteroLinear(self.output_dim, self.output_dim)  # 添加一个线性层，用于将最后的特征进行线性变换

        self.act = thnn.LeakyReLU(0.2)  # LeakyReLU 激活函数

    #
    def forward(self, blocks, features):  # 前向传播
        h = features
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.dropout(h)

        h = self.act(F.normalize(self.linear(h), p=2, dim=1))

        return h
