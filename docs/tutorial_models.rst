Tutorial to Generate Statistical Models
=======================================
In this tutorial we explore how to create and train statistical models to predict
molecular properties using the Pytorch_ library. We will use smiles_ to represent the molecules
and use the csv_ file format to manipulate the molecules and their properties.

As an example, we will predict the `activity coefficient_` for a subset of carboxylic acids taken
from the `GDB-13 database_`. Firstly, We randomly takes a 1000 smiles_ from the database and
compute the `activity coefficient_` using the `COSMO approach_`. We store the values in the `thousand.csv_`
file.

A peek into the file will show you something like: ::

  smiles,E_solv,gammas
  OC(=O)C1OC(C#C)C2NC1C=C2,-11.05439751550119,8.816417146193844
  OC(=O)C1C2NC3C(=O)C2CC13O,-8.98188869016993,52.806217658944995
  OC(=O)C=C(C#C)C1NC1C1CN1,-11.386853547889574,6.413128231164093
  OC(=O)C1=CCCCC2CC2C#C1,-10.578966144649726,1.426566948888662

Where the first column contains the index of the row, the second the solvation energy and finally the
`activity coefficients_` denoted as *gammas*. Once we have the data we can start exploring different statistical methods.

`swan` offers a thin interface to Pytorch_. It takes yaml_ file as input and either train an statistical model or
generates a prediction using a previously trained model. Let's briefly explore the `swan` input.

Simulation input
****************
A typical `swan` input file looks like: ::

  dataset_file:
    tests/test_files/thousand.csv
  properties:
    - gammas

  use_cuda: True

  featurizer:
    fingerprint: atompair

  model:
    name: FingerprintFullyConnected
    parameters:
       input_features: 2048  # Fingerprint size
       hidden_cells: 200
       output_features: 1  # We are predicting a single property 

  torch_config:
    epochs: 100
    batch_size: 100
    optimizer:
      name: sgd
      lr: 0.002

   
**dataset_file**: A csv_ file with the smiles_ and other molecular properties.

**properties**: the columns names of hte csv_ file representing the molecular properties to fit.

**featurizer**: The type of transformation to apply to the smiles_ to generates the features_. Could be either **fingerprint** or **graph**.

Have a look at the  :ref:`available models`.

Training a model
****************
In order to run the training, run the following command: ::

  modeller --mode train -i input.yml

`swan` will generate a log file called  `output.log` with a timestamp for the different steps during the training.
Finally, you can see in your `cwd` a folder called *swan_models* containing the parameters of your statistical model.

It is possible to restart the training procedure by providing the ``--restart`` option like: ::

    modeller --mode train -i input.yml --restart

Predicting new data
*******************
To predict new data you need to provide some smiles for which you want to compute the properties of interest, in this
case the `activity coefficient_`. For doing so, you need to provide in the `dataset_file` entry of the *input.yml*
file the path to a csv_ file containing the smiles, like the `smiles.csv_`: ::

  ,smiles
  0,OC(=O)C1CNC2C3C4CC2C1N34
  1,OC(=O)C1CNC2COC1(C2)C#C
  2,OC(=O)CN1CC(=C)C(C=C)C1=N

Then run the command: ::

  modeler --mode predict -i input.yml

`swan` will look for a *swan_model.pt* file with the previously trained model and will load it.

Finally, you will find a file called "predicted.csv" with the predicted values for the activity coefficients.

..  _deepchem: https://deepchem.io/
.. _smiles: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system
.. _activity coefficient: https://en.wikipedia.org/wiki/Activity_coefficient
.. _GDB-13 database_`: https://pubs.acs.org/doi/abs/10.1021/ja902302h
.. _COSMO approach: https://www.scm.com/doc/ADF/Input/COSMO.html
.. _thousand.csv: https://github.com/nlesc-nano/swan/blob/master/tests/test_files/thousand.csv
.. _features: https://en.wikipedia.org/wiki/Feature_(machine_learning)
.. _smiles.csv: https://github.com/nlesc-nano/swan/blob/master/tests/test_files/smiles.csv
.. _yaml: https://yaml.org
.. _csv: https://en.wikipedia.org/wiki/Comma-separated_values
.. _Pytorch: https://pytorch.org
