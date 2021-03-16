import torch
import torch_geometric as tg
from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from flamingo.features.featurizer import (NUMBER_ATOMIC_GRAPH_FEATURES,
                                          NUMBER_BOND_GRAPH_FEATURES)
from torch_scatter import scatter_add, segment_add_coo


class InvariantPolynomial(torch.nn.Module):
    def __init__(self, irreps_out: str = "0e", lmax: int = 3) -> None:
        super().__init__()

        # Different bond features
        self.num_bond_features = NUMBER_BOND_GRAPH_FEATURES
        # Irreducible representation of the bonds
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)

        # Edges and node attributes are scalars
        representation_bonds = [(1, "0e") for _ in range(NUMBER_BOND_GRAPH_FEATURES)]
        representation_atoms = f"{NUMBER_ATOMIC_GRAPH_FEATURES}x0e"

        # Tensor product the edge attributes with the harmonic basis
        # self.mul_edges = FullTensorProduct(representation_bonds, self.irreps_sh)
        self.mul_edges = TensorProduct(
            representation_bonds,
            self.irreps_sh,
            [(1, ir) for _, ir in self.irreps_sh],
            [(0, l, l, "uvu", False) for l in range(lmax + 1)]
        )

        # middle layer
        # irreps_mid1 = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_mid1 = o3.Irreps("64x0e + 24x1e + 24x1o + 16x2e + 16x2o")
        irreps_mid2 = o3.Irreps("8x0e + 4x1e + 4x1o + 2x2e + 2x2o")

        # Output representation
        irreps_out = o3.Irreps(irreps_out)

        self.tp1 = FullyConnectedTensorProduct(
            irreps_in1=representation_atoms,
            irreps_in2=self.mul_edges.irreps_out,
            irreps_out=irreps_mid1,
        )

        self.tp2 = FullyConnectedTensorProduct(
            irreps_in1=irreps_mid1,
            irreps_in2=self.mul_edges.irreps_out,
            irreps_out=irreps_out,
        )

        # self.tp3 = FullyConnectedTensorProduct(
        #     irreps_in1=irreps_mid2,
        #     irreps_in2=self.mul_edges.irreps_out,
        #     irreps_out=irreps_out
        # )

    def forward(self, data: tg.data.Dataset) -> torch.Tensor:
        # Vector defining the edges
        edge_src, edge_dst = data.edge_index
        edge_vec = data.positions[edge_src] - data.positions[edge_dst]

        # Spherical harmonics
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, normalize=False, normalization='component')

        # Edge attributes in the harmonic basis
        edge_attr = self.mul_edges(data.edge_attr, edge_sh)

        # Count the number of neighbors for each atom
        num_of_bonds_per_atom = tg.utils.degree(edge_dst).unsqueeze(-1)

        # For each edge, tensor product the node attributes with the
        # edge features in the spherical harmonics
        edge_features = self.tp1(data.x[edge_src], edge_attr)
        node_features = scatter_add(edge_features, edge_dst, dim=0)

        # Normalize by the number of bonds for each atom
        node_features = node_features.div(num_of_bonds_per_atom ** 0.5)

        # communicate information to the neighbors
        edge_features = self.tp2(node_features[edge_src], edge_attr)
        node_features = scatter_add(edge_features, edge_dst, dim=0)

        # # Normalize by the number of bonds for each atom
        # node_features = node_features.div(num_of_bonds_per_atom ** 0.5)

        # # communicate information to the neighbors
        # edge_features = self.tp3(node_features[edge_src], edge_attr)
        # node_features = scatter_add(edge_features, edge_dst, dim=0)

        # Normalize by the number of bonds for each atom
        node_features = node_features.div(num_of_bonds_per_atom ** 0.5)

        # Sum for each molecule
        acc = segment_add_coo(node_features, data.batch)

        # Count how many bonds are in a given batch
        atoms_per_molecule_in_batch = tg.utils.degree(data.batch).unsqueeze(-1)

        # Normalize by the number of atoms beloging to a given molecule
        return acc.div(atoms_per_molecule_in_batch ** 0.5)
