# Machine learning model to predict CDFT properties

The following table shows the **mean rvalue** resulting from the linear regression
between the predicted as function of the ground true labels. The mean values where
computing with the results of **3** training/validation datasets picked randomly
in a 0.8/0.2 proportion, respectively.

The hyperparameters use to perform the training can be found at [model's hyperparameters](Training_hyperparameters.md).

## Amines

**Number of molecules in dataset: 6277**

|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.87   | 0.70
| Dissociation energy (electrofuge)  | 0.82	  | 0.56
| Electroaccepting power(w+)         | 0.71	  | 0.46
| Electrodonating power (w-)         | 0.79	  | 0.64
| Electronegativity (chi=-mu)        | 0.89	  | 0.57
| Electronic chemical potential (mu) | 0.85	  | 0.53
| Electronic chemical potential (mu+)| 0.85	  | 0.28
| Electronic chemical potential (mu-)| 0.89	  | 0.83
| Electrophilicity index (w=omega)   | 0.70	  | 0.57
| Global Dual Descriptor Deltaf+     | 0.65	  | 0.27
| Global Dual Descriptor Deltaf-     | 0.67	  | 0.31
| Hardness (eta)                     | 0.86	  | 0.62
| Hyperhardness (gamma)              | 0.83	  | 0.45
| Net Electrophilicity               | 0.66	  | 0.59
| Softness (S)                       | 0.88	  | 0.14





## Carboxylic acids


|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 
| Dissociation energy (electrofuge)  | 
| Electroaccepting power(w+)         | 
| Electrodonating power (w-)         | 
| Electronegativity (chi=-mu)        | 
| Electronic chemical potential (mu) | 
| Electronic chemical potential (mu+)| 
| Electronic chemical potential (mu-)| 
| Electrophilicity index (w=omega)   | 
| Global Dual Descriptor Deltaf+     | 
| Global Dual Descriptor Deltaf-     | 
| Hardness (eta)                     | 
| Hyperhardness (gamma)              | 
| Net Electrophilicity               | 
| Softness (S)                       |
