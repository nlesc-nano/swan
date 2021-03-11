.. _available models:

Available models
================
Currently **Swan** Implements the following models:

Fully Connected Neural Network
******************************
A standard fully connected neural network that takes *fingerprints* as
input features. To use the model you need to specify in the ``model`` section
of the input YAML file the following: ::

  model:
    name: FingerprintFullyConnected
    parameters:
      input_features: 2048
      hidden_cells: 100
      output_features: 1

The model takes 3 additional optional parameters:
* ``input_features``: fingerprint size. Default 2048.
* ``hidden_cells``: Hiden number of cell(or nodes). Default 100.
* ``num_labels``: the amount of labels to predict. Default 1.

Also, the model requires as a ``featurizer`` a fingerprint calculator that can be provided like: ::

  featurizer:
    fingerprint: atompair

Available fingerprints algorithms are: ``atompair`` (default), ``morgan`` or ``torsion``. These
algorithms are provided by `RDKIT descriptor package <https://rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html>`_.


Message Passing Neural Network
******************************
Implementation of the message passing neural network (MPNN) reported at `<https://arxiv.org/abs/1704.01212>`_.
If you don't have an idea what a MPNN is have a look at
`this introduction to Graph Neural Networks <https://www.youtube.com/watch?v=zCEYiCxrL_0&list=PLVqPBNulzDDg8ieQZ2G643UFbHm-qWW7Z&index=1&t=2239s>`_.

To train your model using the MPNN you need to provide the following section in the YAML input file: ::

  model:
    name: MPNN
    parameters:
      dim: 10
      num_labels: 1
      batch_size: 128
      num_iterations: 3

The optional parameters for the model are: ::
* ``dim``
* ``num_labels``: the amount of labels to predict. Default 1.
*  ``batch_size``: the size of the batch used to train the model. Default 128
* ``num_iterations``: number of steps to interchange messages for each epoch. Default 3.

Additionally the model requires the use of the following featurizer: ::

  featurizer:
   graph: molecular
   file_geometries: geometries.json

Where ``file_geometries`` is a JSON file containing an array of molecules on PDB format. Check
`the example file <https://github.com/nlesc-nano/swan/blob/main/tests/files/cdft_geometries.json>`_
If the ``file_geometries`` is not set in the input the model will try to use the RDKit geometries.
