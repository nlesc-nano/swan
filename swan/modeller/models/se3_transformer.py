
"""See: https://github.com/FabianFuchsML/se3-transformer-public"""

import torch
from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import (GAvgPooling, GConvSE3, GMaxPooling,
                                           GNormSE3, GSE3Res, get_basis_and_r)

from swan.dataset.features.featurizer import (NUMBER_ATOMIC_GRAPH_FEATURES,
                                              NUMBER_BOND_GRAPH_FEATURES)

__all__ = ["TFN", "SE3Transformer"]


class TFN(torch.nn.Module):
    """SE(3) equivariant GCN"""
    def __init__(self, num_layers: int, num_channels: int, num_nlayers: int = 1, num_degrees: int = 4):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.num_channels_out = num_channels * num_degrees
        self.edge_dim = NUMBER_BOND_GRAPH_FEATURES

        self.fibers = {'in': Fiber(1, NUMBER_ATOMIC_GRAPH_FEATURES),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        blocks = self._build_gcn(self.fibers, 1)
        self.block0, self.block1, self.block2 = blocks
        print(self.block0)
        print(self.block1)
        print(self.block2)

    def _build_gcn(self, fibers, out_dim):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers - 1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        block1 = [GMaxPooling()]

        block2 = []
        block2.append(torch.nn.Linear(self.num_channels_out, self.num_channels_out))
        block2.append(torch.nn.ReLU(inplace=True))
        block2.append(torch.nn.Linear(self.num_channels_out, out_dim))

        return torch.nn.ModuleList(block0), torch.nn.ModuleList(block1), torch.nn.ModuleList(block2)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.block1:
            h = layer(h, G)

        for layer in self.block2:
            h = layer(h)

        return h


class SE3Transformer(torch.nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int,
                 num_channels: int, num_nlayers: int = 1, num_degrees: int = 4,
                 div: float = 4, pooling: str = 'avg', n_heads: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = NUMBER_BOND_GRAPH_FEATURES
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, NUMBER_ATOMIC_GRAPH_FEATURES),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees * self.num_channels)}

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = [
            torch.nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(self.fibers['out'].n_features, out_dim)
        ]

        return torch.nn.ModuleList(Gblock), torch.nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h
