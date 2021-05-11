# Machine learning model to predict CDFT properties

The following table shows the **mean rvalue** resulting from the linear regression
between the predicted as function of the ground true labels. Therefore, the closer
these values are to 1 the better the better the model is to predict the properties *with
the available amount of training data*.

The mean values where computing with the results of **5** training/validation datasets picked randomly
in a 0.8/0.2 proportion, respectively.

The hyperparameters use to perform the training can be found at [model's hyperparameters](Training_hyperparameters.md).


## Amines
The [amines CDFT dataset](https://github.com/nlesc-nano/swan/blob/main/data/Amines/CDFT/all_amines.csv) has the following properties:
* Number of molecules in dataset: **3109**
* Number of labels: **15**

The [geometries file](https://github.com/nlesc-nano/swan/blob/main/data/Amines/CDFT/all_amines.csv) contains all the optimized geometries for the previous dataset.


|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.73   | 0.70   | 0.78 |
| Dissociation energy (electrofuge)  | 0.39	  | 0.32   | 0.39 |
| Electroaccepting power(w+)         | 0.42	  | 0.36   | 0.55 |
| Electrodonating power (w-)         | 0.52	  | 0.54   | 0.66 |
| Electronegativity (chi=-mu)        | 0.54	  | 0.50   | 0.61 |
| Electronic chemical potential (mu) | 0.54	  | 0.53   | 0.59 |
| Electronic chemical potential (mu+)| 0.38	  | 0.25   | 0.34 |
| Electronic chemical potential (mu-)| 0.78	  | 0.81   | 0.85 |
| Electrophilicity index (w=omega)   | 0.51	  | 0.53   | 0.62 |
| Global Dual Descriptor Deltaf+     | 0.33	  | 0.26   | 0.35 |
| Global Dual Descriptor Deltaf-     | 0.35	  | 0.26   | 0.36 |
| Hardness (eta)                     | 0.53	  | 0.53   | 0.56 |
| Hyperhardness (gamma)              | 0.39	  | 0.29   | 0.41 |
| Net Electrophilicity               | 0.47	  | 0.55   | 0.62 |
| Softness (S)                       | 0.42	  | 0.07   | 0.20 |



## Carboxylic acids
The [carboxylic acids CDFT dataset](https://github.com/nlesc-nano/swan/blob/main/data/Carboxylic_acids/CDFT/all_carboxylics.csv) has the following properties:
* Number of molecules in dataset: **11044**
* Number of labels: **15**

The [geometries file](https://github.com/nlesc-nano/swan/blob/main/data/Carboxylic_acids/CDFT/all_geometries_carboxylics.json) contains all the optimized geometries for the previous dataset.


|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.85   | 0.85 |  0.87 |
| Dissociation energy (electrofuge)  | 0.75   | 0.83 | 	0.83 |
| Electroaccepting power(w+)         | 0.63	  | 0.64 | 	0.66 |
| Electrodonating power (w-)         | 0.79	  | 0.79 | 	0.84 |
| Electronegativity (chi=-mu)        | 0.81	  | 0.85 | 	0.88 |
| Electronic chemical potential (mu) | 0.81	  | 0.80 | 	0.88 |
| Electronic chemical potential (mu+)| 0.72	  | 0.78 | 	0.77 |
| Electronic chemical potential (mu-)| 0.86	  | 0.86 | 	0.91 |
| Electrophilicity index (w=omega)   | 0.72	  | 0.75 | 	0.79 |
| Global Dual Descriptor Deltaf+     | 0.59	  | 0.59 | 	0.61 |
| Global Dual Descriptor Deltaf-     | 0.59	  | 0.59 | 	0.60 |
| Hardness (eta)                     | 0.80	  | 0.74 | 	0.83 |
| Hyperhardness (gamma)              | 0.67	  | 0.68 | 	0.70 |
| Net Electrophilicity               | 0.72	  | 0.78 | 	0.80 |
| Softness (S)                       | 0.41	  | 0.57 | 	0.44 |

