# Change Log

# 0.3.0 [11/08/2020]

## New
* Introduce Pipeline to filter ligands (#26)
* Use [Horovod](https://github.com/horovod/horovod) to distribute the training 

# 0.2.0 [25/02/2020]

## New
* Allow to  train in **GPU**.
* Use [Pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) to create molecular graph convolutional networks.

## Changed
* Replace [deepchem](https://deepchem.io/) with [pytorch](https://pytorch.org)
* Use [rdkit](https://www.rdkit.org) to compute the fingerprints.

# 0.1.0

## New

* Interface to [deepchem](https://deepchem.io/) to generate regression models.
* Interface to [CAT](https://github.com/nlesc-nano/CAT) to compute solvation energies and activity coefficients.
* Train or predict values for a set of smiles using a statistical model.
