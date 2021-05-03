# Hyperparameters used to train the NN

The hyperparameters are taken from the following references:

* [Simple Feedforward NN](https://arxiv.org/abs/1509.09292) (No convolution is performed)
* [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
* [SE(3)-Transformer](https://arxiv.org/abs/2006.10503)

## FullyConnectedFingerPrint
| Parameter       | value |
| :-------------: | :---: |
| num_labels      | 1     |
| num_iterations  | 3     |
| output_channels | 10    |
| batch_size      | 20    |
| learning rate   | 5e-4  |


## MPNN

|     Parameter   | value |
| :-------------: | :---: |
| num_labels      | 1     |
| num_iterations  | 3     |
| output_channels | 10    |
| batch_size      | 20    |
| learning rate   | 5e-4  |


## SE3Transformer

| Parameter       | value  |
| :-------------: | :----: |
| num_layers      | 4      |
| num_channels    | 16     |
| num_nlayers     | 0      |
| num_degrees     | 4      |
| div             | 4      |
| pooling         | 'avg'  |
| n_heads         | 1      |
| batch_size      | 32     |
| learning rate   | 1e-3   |
