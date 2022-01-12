from pathlib import Path

import pandas as pd
import torch

from swan.dataset import FingerprintsData

path_files = Path("data")

# Compute full dataset of fingerprints and all properties, for both ligands
paths = {
    'carboxylics': path_files / 'Carboxylic_acids/CDFT/all_carboxylics.csv',
    'amines': path_files /  'Amines/CDFT/all_amines.csv'
}
frames = {ligand_type: pd.read_csv(path, index_col='Unnamed: 0') for ligand_type, path in paths.items()}
properties = list(frames['carboxylics'].columns[2:])
fp_data = {ligand_type: FingerprintsData(path, sanitize=False, properties=properties) 
           for ligand_type, path in paths.items()}
Xs = {ligand_type: fp.fingerprints for ligand_type, fp in fp_data.items()}
ys = {ligand_type: fp.labels for ligand_type, fp in fp_data.items()}

# shuffle data for both ligands
torch.manual_seed(42)  # to make it deterministic
indices = {ligand_type: torch.randperm(X.shape[0]) for ligand_type, X in Xs.items()}
Xs_shuffled = {ligand_type: X[indices] for (ligand_type, X), indices in zip(Xs.items(), indices.values())}
ys_shuffled = {ligand_type: y[indices] for (ligand_type, y), indices in zip(ys.items(), indices.values())}

# set aside 1000 data points of carboxylics as the test set
n_test = 1_000
test_data = Xs_shuffled['carboxylics'][:n_test], ys_shuffled['carboxylics'][:n_test]

# the remaining carboxylics, in addition to potentially all amines, are the training set
# these are to be split into training and validation sets during usage
train_data_carboxylics = Xs_shuffled['carboxylics'][n_test:], ys_shuffled['carboxylics'][n_test:]
train_data_amines = Xs_shuffled['amines'], ys_shuffled['amines']

# save
torch.save(test_data, path_files / 'Carboxylic_acids/CDFT/fingerprint_test')
torch.save(train_data_carboxylics, path_files / 'Carboxylic_acids/CDFT/fingerprint_train')
torch.save(train_data_amines, path_files / 'Amines/CDFT/fingerprint_train')