import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# from net.utils.tgcn import ConvTemporalGraphical
# from net.utils.graph import Graph
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, graph=None,
                 **kwargs):
        super().__init__()

        # load graph
        if graph is None:
            self.graph = Graph(**graph_args)
        else:
            Graph_ic = import_class(graph)
            self.graph = Graph_ic(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        # self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        self.fcn = nn.Linear(256, num_class)

        self.apply(weights_init)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, C, T, V = x.size()
        features = x.view(N, M, C, T, V)
        features = features.permute(0, 2, 1, 3, 4).contiguous()  # N, c, M, t, v

        local_features = features.permute(0, 2, 1, 3, 4).contiguous()  # N, M, C, T, V
        local_features = local_features.reshape(N * M, C, T, V)  # N * M, C, T, V
        local_features = F.avg_pool2d(local_features, (T, V))  # N * M, C, 1, 1
        local_features = local_features.reshape(N, M, C)  # N, M, C

        global_features = local_features.clone()
        global_features = global_features.mean(dim=1)  # N, C
        x = self.fcn(global_features)
        return x

    def get_features(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, C, T, V = x.size()
        features = x.view(N, M, C, T, V)
        features = features.permute(0, 2, 1, 3, 4).contiguous()  # N, c, M, t, v

        local_features = features.permute(0, 2, 1, 3, 4).contiguous()  # N, M, C, T, V
        local_features = local_features.reshape(N * M, C, T, V)  # N * M, C, T, V
        local_features = F.avg_pool2d(local_features, (T, V))  # N * M, C, 1, 1
        local_features = local_features.reshape(N, M, C)  # N, M, C

        global_features = local_features.clone()
        global_features = global_features.mean(dim=1)  # N, C

        return global_features

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        local_ft = x.view(N, M, c, t, v)
        local_ft = local_ft.permute(0, 2, 1, 3, 4).contiguous()  # N, c, M, t, v

        return local_ft

    def extract_feature_person(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        # local_ft = x.view(N, M, c, t, v)
        local_ft = x.reshape(N * M, c, t, v)  # N * M, C, T, V
        local_ft = F.avg_pool2d(local_ft, (t, v))  # N*M, C, 1, 1
        local_ft = local_ft.reshape(N, M, c)  # N, M, C

        return local_ft

    def visualization(self, x):
        with torch.no_grad():
            object_ft = self.extract_feature_person(x)
            sample_ft = object_ft.mean(dim=1)
            object_ft = F.normalize(object_ft, dim=2)
            sample_ft = F.normalize(sample_ft, dim=1)
        return object_ft, sample_ft


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                        11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6),
                             (6, 7), (1, 14), (14, 8), (8, 9), (9, 10),
                             (14, 11), (11, 12), (12, 13)]
            self.edge = self_link + neighbor_link
            self.center = 1
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


class MutualGraph():
    def __init__(self, dataset, graph, labeling, num_person_out=1, max_hop=10, dilation=1, normalize=True,
                 threshold=0.2, **kwargs):
        self.dataset = dataset
        self.labeling = labeling
        self.graph = graph
        self.normalize = normalize
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_person_out = num_person_out
        self.threshold = threshold

        # get edges
        self.num_node, self.edge, self.connect_joint, self.parts, self.center = self._get_edge()

        # get adjacency matrix
        self.A = self._get_adjacency()

    def __str__(self):
        return self.A

    def _get_edge(self):
        if self.dataset == 'kinetics':
            num_node = 18
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14), (8, 11)]
            connect_joint = np.array([1, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 5, 11, 12, 0, 0, 14, 15])
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([11, 12, 13]),  # left_leg
                np.array([8, 9, 10]),  # right_leg
                np.array([0, 1, 14, 15, 16, 17])  # torso
            ]
            center = 1
        elif self.dataset in ['ntu', 'ntu120', 'ntu_mutual', 'ntu120_mutual', 'ntu_original']:
            if self.graph == 'physical':
                num_node = 25
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                  (22, 23), (23, 8), (24, 25), (25, 12)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array(
                    [2, 2, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 23, 8, 25, 12]) - 1
                parts = [
                    np.array([5, 6, 7, 8, 22, 23]) - 1,  # left_arm
                    np.array([9, 10, 11, 12, 24, 25]) - 1,  # right_arm
                    np.array([13, 14, 15, 16]) - 1,  # left_leg
                    np.array([17, 18, 19, 20]) - 1,  # right_leg
                    np.array([1, 2, 3, 4, 21]) - 1  # torso
                ]
                center = 21 - 1
            elif self.graph == 'mutual':
                num_node = 50
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                  (22, 23), (23, 8), (24, 25), (25, 12)] + \
                                 [(26, 27), (27, 46), (28, 46), (29, 28), (30, 46),
                                  (31, 30), (32, 31), (33, 32), (34, 46), (35, 34),
                                  (36, 35), (37, 36), (38, 26), (39, 38), (40, 39),
                                  (41, 40), (42, 26), (43, 42), (44, 43), (45, 44),
                                  (47, 48), (48, 33), (49, 50), (50, 37)] + \
                                 [(21, 46)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array(
                    [1, 1, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 22, 7, 24, 11, 26, 26, 45,
                     27, 45, 29, 30, 31, 45, 33, 34, 35, 25, 37, 38, 39, 25, 41, 42, 43, 26, 47, 32, 49, 36])
                parts = [
                    # left_arm
                    np.array([5, 6, 7, 8, 22, 23]) - 1,
                    np.array([5, 6, 7, 8, 22, 23]) + 25 - 1,
                    # right_arm
                    np.array([9, 10, 11, 12, 24, 25]) - 1,
                    np.array([9, 10, 11, 12, 24, 25]) + 25 - 1,
                    # left_leg
                    np.array([13, 14, 15, 16]) - 1,
                    np.array([13, 14, 15, 16]) + 25 - 1,
                    # right_leg
                    np.array([17, 18, 19, 20]) - 1,
                    np.array([17, 18, 19, 20]) + 25 - 1,
                    # torso
                    np.array([1, 2, 3, 4, 21]) - 1,
                    np.array([1, 2, 3, 4, 21]) + 25 - 1
                ]
                center = 21 - 1
            elif self.graph == 'mutual-inter':
                num_node = 50
                neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                                  (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                                  (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                                  (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                                  (22, 23), (23, 8), (24, 25), (25, 12)] + \
                                 [(26, 27), (27, 46), (28, 46), (29, 28), (30, 46),
                                  (31, 30), (32, 31), (33, 32), (34, 46), (35, 34),
                                  (36, 35), (37, 36), (38, 26), (39, 38), (40, 39),
                                  (41, 40), (42, 26), (43, 42), (44, 43), (45, 44),
                                  (47, 48), (48, 33), (49, 50), (50, 37)] + \
                                 [(21, 46)] + \
                                 [(23, 25), (48, 50), (23, 48), (25, 50)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array(
                    [1, 1, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 22, 7, 24, 11, 26, 26, 45,
                     27, 45, 29, 30, 31, 45, 33, 34, 35, 25, 37, 38, 39, 25, 41, 42, 43, 26, 47, 32, 49, 36])
                parts = [
                    # left_arm
                    np.array([5, 6, 7, 8, 22, 23]) - 1,
                    np.array([5, 6, 7, 8, 22, 23]) + 25 - 1,
                    # right_arm
                    np.array([9, 10, 11, 12, 24, 25]) - 1,
                    np.array([9, 10, 11, 12, 24, 25]) + 25 - 1,
                    # left_leg
                    np.array([13, 14, 15, 16]) - 1,
                    np.array([13, 14, 15, 16]) + 25 - 1,
                    # right_leg
                    np.array([17, 18, 19, 20]) - 1,
                    np.array([17, 18, 19, 20]) + 25 - 1,
                    # torso
                    np.array([1, 2, 3, 4, 21]) - 1,
                    np.array([1, 2, 3, 4, 21]) + 25 - 1
                ]
                center = 21 - 1
        elif self.dataset == 'sbu':
            if self.graph == 'physical':
                num_node = 15
                neighbor_1base = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7),
                                  (7, 8), (8, 9), (3, 10), (10, 11), (11, 12),
                                  (3, 13), (13, 14), (14, 15)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array([2, 3, 3, 3, 4, 5, 3, 7, 8, 3, 10, 11, 3, 13, 14]) - 1
                parts = [
                    # left_arm
                    np.array([4, 5, 6]) - 1,
                    # right_arm
                    np.array([7, 8, 9]) - 1,
                    # left_leg
                    np.array([10, 11, 12]) - 1,
                    # right_leg
                    np.array([13, 14, 15]) - 1,
                    # torso
                    np.array([1, 2, 3]) - 1,
                ]
                center = 3 - 1
            elif self.graph == 'mutual':
                num_node = 30
                neighbor_1base = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 7),
                                  (7, 8), (8, 9), (3, 10), (10, 11), (11, 12),
                                  (3, 13), (13, 14), (14, 15)]
                neighbor_1base += [(i + 15, j + 15) for (i, j) in neighbor_1base] + [(2, 2 + 15)]
                neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
                connect_joint = np.array([2, 3, 3, 3, 4, 5, 3, 7, 8, 3, 10, 11, 3, 13, 14]) - 1
                parts = [
                    # left_arm
                    np.array([4, 5, 6]) - 1,
                    np.array([4, 5, 6]) + 15 - 1,
                    # right_arm
                    np.array([7, 8, 9]) - 1,
                    np.array([7, 8, 9]) + 15 - 1,
                    # left_leg
                    np.array([10, 11, 12]) - 1,
                    np.array([10, 11, 12]) + 15 - 1,
                    # right_leg
                    np.array([13, 14, 15]) - 1,
                    np.array([13, 14, 15]) + 15 - 1,
                    # torso
                    np.array([1, 2, 3]) - 1,
                    np.array([1, 2, 3]) + 15 - 1,
                ]
                center = 3 - 1
        elif self.dataset == 'volleyball':
            num_node = 25
            neighbor_link = [(0, 1), (0, 15), (0, 16), (15, 17), (16, 18),
                             (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                             (1, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 22), (22, 23),
                             (8, 12), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)
                             ]
            connect_joint = np.array(
                [1, 1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 10, 8, 12, 13, 0, 0, 15, 16, 14, 19, 14, 11, 22, 11])
            parts = [
                np.array([5, 6, 7]),  # left_arm
                np.array([2, 3, 4]),  # right_arm
                np.array([9, 10, 11, 22, 23, 24]),  # left_leg
                np.array([12, 13, 14, 19, 20, 21]),  # right_leg
                np.array([0, 1, 8, 15, 16, 17, 18])  # torso
            ]
            center = 1
            if self.graph == 'multi-person':
                neighbor_link_nperson = []
                connect_joint_nperson = []
                parts_nperson = []
                for i in range(self.num_person_out):
                    for x in connect_joint:
                        connect_joint_nperson.append(x + i * num_node)
                    for x, y in neighbor_link:
                        neighbor_link_nperson.append((x + i * num_node, y + i * num_node))
                    for p in range(len(parts)):
                        parts_nperson.append(parts[p] + i * num_node)
                num_node *= self.num_person_out

                neighbor_link = neighbor_link_nperson
                connect_joint = connect_joint_nperson
                parts = parts_nperson
        else:
            # logging.info('')
            # logging.error('Error: Do NOT exist this dataset: {}!'.format(self.dataset))
            raise ValueError()
        self_link = [(i, i) for i in range(num_node)]
        edge = self_link + neighbor_link
        return num_node, edge, connect_joint, parts, center

    def _get_hop_distance(self):
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            A[j, i] = 1
            A[i, j] = 1
        self.oA = A
        hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def _get_adjacency(self):

        if self.labeling == 'distance':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'spatial':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if hop_dis[j, i] == hop:
                            # if hop_dis[j, self.center] == np.inf or hop_dis[i, self.center] == np.inf:
                            #     continue
                            if hop_dis[j, self.center] == hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif hop_dis[j, self.center] > hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)

        elif self.labeling == 'zeros':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))

        elif self.labeling == 'ones':
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.ones((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(A[i])


        elif self.labeling == 'eye':

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i in range(len(valid_hop)):
                A[i] = self._normalize_digraph(np.eye(self.num_node, self.num_node))

        elif self.labeling == 'pairwise0':
            # pairwise0: only pairwise inter-body link
            assert 'mutual' in self.graph

            valid_hop = range(0, self.max_hop + 1, self.dilation)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            v = self.num_node // 2
            for i in range(len(valid_hop)):
                A[i, v:, :v] = np.eye(v, v)
                A[i, :v, v:] = np.eye(v, v)
                A[i] = self._normalize_digraph(A[i])


        elif self.labeling == 'pairwise1':
            assert 'mutual' in self.graph
            v = self.num_node // 2
            self.edge += [(i, i + v) for i in range(v)]
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1
            normalize_adjacency = self._normalize_digraph(adjacency)
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        elif self.labeling == 'geometric':
            hop_dis = self._get_hop_distance()
            valid_hop = range(0, self.max_hop + 1, self.dilation)
            adjacency = np.zeros((self.num_node, self.num_node))
            for hop in valid_hop:
                adjacency[hop_dis == hop] = 1

            geometric_matrix = np.load(os.path.join(os.getcwd(), 'src/dataset/a.npy'))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if geometric_matrix[i, j] > self.threshold:
                        adjacency[i, j] += geometric_matrix[i, j]
            normalize_adjacency = self._normalize_digraph(adjacency)

            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][hop_dis == hop] = normalize_adjacency[hop_dis == hop]

        return A

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

class ConvTemporalGraphical(nn.Module):

    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)