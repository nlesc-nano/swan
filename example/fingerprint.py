from swan.modeller import FingerprintModeller
from swan.dataset import FingerprintsDataset
from mynet import FingerprintFullyConnected

# define the dataset
path = '../data/Carboxylic_acids/GDB13/Results/CDFT/cdft.csv'
dataset = FingerprintsDataset(path, properties=['Hyperhardness (gamma)'])

# define the network
net = FingerprintFullyConnected()

# define the modeller
model = FingerprintModeller(net, dataset)
