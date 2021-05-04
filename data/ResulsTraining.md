# Machine learning model to predict CDFT properties

The following table shows the **mean rvalue** resulting from the linear regression
between the predicted as function of the ground true labels. Therefore, the closer
these values are to 1 the better the better the model is to predict the properties *with
the available amount of training data*.

The mean values where computing with the results of **3** training/validation datasets picked randomly
in a 0.8/0.2 proportion, respectively.

The hyperparameters use to perform the training can be found at [model's hyperparameters](Training_hyperparameters.md).


## Amines

**Number of molecules in dataset: 6277**

|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.87   | 0.70  | 0.85 |
| Dissociation energy (electrofuge)  | 0.82	  | 0.56  | 0.62 |
| Electroaccepting power(w+)         | 0.71	  | 0.46  | 0.52 |
| Electrodonating power (w-)         | 0.79	  | 0.64  | 0.71 |
| Electronegativity (chi=-mu)        | 0.89	  | 0.57  | 0.74 |
| Electronic chemical potential (mu) | 0.85	  | 0.53  | 0.75 |
| Electronic chemical potential (mu+)| 0.85	  | 0.28  | 0.69 |
| Electronic chemical potential (mu-)| 0.89	  | 0.83  | 0.87 |
| Electrophilicity index (w=omega)   | 0.70	  | 0.57  | 0.74 |
| Global Dual Descriptor Deltaf+     | 0.65	  | 0.27  | 0.54 |
| Global Dual Descriptor Deltaf-     | 0.67	  | 0.31  | 0.55 |
| Hardness (eta)                     | 0.86	  | 0.62  | 0.79 |
| Hyperhardness (gamma)              | 0.83	  | 0.45  | 0.62 |
| Net Electrophilicity               | 0.66	  | 0.59  | 0.74 |
| Softness (S)                       | 0.88	  | 0.14  | 0.54 |




**Number of molecules in dataset: 14334**


## Carboxylic acids


|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    |  0.87  | 0.85  | 0.90 | 
| Dissociation energy (electrofuge)  | 	0.79  | 0.82  | 0.80 | 
| Electroaccepting power(w+)         | 	0.68  | 0.69  | 0.75 | 
| Electrodonating power (w-)         | 	0.82  | 0.79  | 0.82 | 
| Electronegativity (chi=-mu)        | 	0.86  | 0.84  | 0.88 | 
| Electronic chemical potential (mu) | 	0.86  | 0.63  | 0.88 | 
| Electronic chemical potential (mu+)| 	0.78  | 0.59  | 0.79 | 
| Electronic chemical potential (mu-)| 	0.89  | 0.85  | 0.91 | 
| Electrophilicity index (w=omega)   | 	0.74  | 0.75  | 0.79 | 
| Global Dual Descriptor Deltaf+     | 	0.65  | 0.64  | 0.63 | 
| Global Dual Descriptor Deltaf-     | 	0.65  | 0.60  | 0.61 | 
| Hardness (eta)                     | 	0.82  | 0.79  | 0.81 | 
| Hyperhardness (gamma)              | 	0.76  | 0.72  | 0.74 | 
| Net Electrophilicity               | 	0.78  | 0.76  | 0.81 | 
| Softness (S)                       |	0.62  | 0.51  | 0.52 |
