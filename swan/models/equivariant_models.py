import torch
import torch_geometric as tg
from e3nn import o3
from e3nn.o3 import FullTensorProduct, FullyConnectedTensorProduct
from flamingo.features.featurizer import (NUMBER_ATOMIC_GRAPH_FEATURES,
                                          NUMBER_BOND_GRAPH_FEATURES)
from torch_scatter import scatter


class InvariantPolynomial(torch.nn.Module):
    def __init__(self, irreps_out: str = "0e", lmax: int = 2) -> None:
        super().__init__()

        # Different bond features
        self.num_bond_features = NUMBER_BOND_GRAPH_FEATURES
        # Irreducible representation of the bonds
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        # Edges and node attributes are scalars
        representation_bonds = [(1, "0e") for _ in range(NUMBER_BOND_GRAPH_FEATURES)]
        representation_atoms = f"{NUMBER_ATOMIC_GRAPH_FEATURES}x0e"

        # Tensor product the edge attributes with the harmonic basis
        self.mul_edges = FullTensorProduct(representation_bonds, self.irreps_sh)

        # middle layer
        irreps_mid = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")

        # Output representation
        irreps_out = o3.Irreps(irreps_out)

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=representation_atoms,
            irreps_in2=self.mul_edges.irreps_out,
            irreps_out=irreps_mid,
        )
        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid,
            irreps_in2=self.mul_edges.irreps_out,
            irreps_out=irreps_out
        )

        self.bn1 = torch.nn.BatchNorm1d(irreps_mid.dim)
        # self.bn2 = torch.nn.BatchNorm1d(irreps_mid.dim)

    def forward(self, data: tg.data.Dataset) -> torch.Tensor:
        # Vector defining the edges
        edge_src, edge_dst = data.edge_index
        edge_vec = data.positions[edge_src] - data.positions[edge_dst]

        # Spherical harmonics
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=False, normalization='component')

        # Edge attributes in the harmonic basis
        edge_attr = self.mul_edges(data.edge_attr, edge_sh)

        # For each edge, tensor product the node features with the
        # edge features in the spherical harmonics
        edge_features = self.tp1(data.x[edge_src], edge_attr)
        node_features = scatter(edge_features, edge_dst, dim=0)

        # communicate information to the neighbors
        edge_features = self.tp2(node_features[edge_src], edge_attr)
        node_features = scatter(edge_features, edge_dst, dim=0)

        return scatter(node_features, data.batch, dim=0)
