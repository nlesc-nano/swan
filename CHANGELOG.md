# Change Log

# 0.6.0 [Unreleased]
## New
* Add interface to scikit regressors (#85)
* Add interface to HDF5 to store the training results (#88)

## Changed
* Fix prediction functionality (#81)

# 0.5.0 [04/05/2021]
## Changed
* Fix graph neural network implementation (#59)
* Rename the graph neural network to MPNN (message passing NN)

## New
* Interface to [se3-transformer](https://www.dgl.ai/pages/start.html) (#57)
* Calculate a guess to the molecular coordinates using a force field optimization
* Add bond distance as feature for the GNN using the optimized geometries (#59)
* Add early stopping functionality

# 0.4.0 [02/10/2020]

## Changed
* Removed duplicate CAT functionality (#36)
* Moved the properties computation and filtering to its own repo(#44, #45)

# 0.3.0 [21/08/2020]

## New
* Introduce Pipeline to filter ligands (#13, #26)
* Use [SCScore](https://pubs.acs.org/doi/10.1021/acs.jcim.7b00622)
* Use [Horovod](https://github.com/horovod/horovod) to distribute the training 
* Add [mypy](http://mypy-lang.org/) test

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
