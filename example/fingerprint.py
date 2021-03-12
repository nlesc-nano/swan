from swan import FingerprintModeller
from swan.dataset import FingerprintsDataset
# from mynet import FingerprintFullyConnected
# import pandas as pd

path = '../data/Carboxylic_acids/GDB13/Results/CDFT/cdft.csv'
dataset = FingerprintsDataset(path, properties='Hyperhardness (gamma)')
model = FingerprintModeller(net, data, opt)