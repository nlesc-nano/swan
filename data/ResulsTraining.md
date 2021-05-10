# Machine learning model to predict CDFT properties

The following table shows the **mean rvalue** resulting from the linear regression
between the predicted as function of the ground true labels. Therefore, the closer
these values are to 1 the better the better the model is to predict the properties *with
the available amount of training data*.

The mean values where computing with the results of **5** training/validation datasets picked randomly
in a 0.8/0.2 proportion, respectively.

The hyperparameters use to perform the training can be found at [model's hyperparameters](Training_hyperparameters.md).


## Amines

**Number of molecules in dataset: 3109**

|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.73   | 0.78   | 0.78 | 
| Dissociation energy (electrofuge)  | 0.39	  | 0.31   | 0.39 | 
| Electroaccepting power(w+)         | 0.42	  | 0.35   | 0.55 | 
| Electrodonating power (w-)         | 0.52	  | 0.53   | 0.66 | 
| Electronegativity (chi=-mu)        | 0.54	  | 0.48   | 0.61 | 
| Electronic chemical potential (mu) | 0.54	  | 0.54   | 0.59 | 
| Electronic chemical potential (mu+)| 0.38	  | 0.24   | 0.34 | 
| Electronic chemical potential (mu-)| 0.78	  | 0.80   | 0.85 | 
| Electrophilicity index (w=omega)   | 0.51	  | 0.53   | 0.62 | 
| Global Dual Descriptor Deltaf+     | 0.33	  | 0.24   | 0.35 | 
| Global Dual Descriptor Deltaf-     | 0.35	  | 0.24   | 0.36 | 
| Hardness (eta)                     | 0.53	  | 0.53   | 0.56 | 
| Hyperhardness (gamma)              | 0.39	  | 0.29   | 0.41 | 
| Net Electrophilicity               | 0.47	  | 0.61   | 0.62 | 
| Softness (S)                       | 0.42	  | 0.06   | 0.20 |


**Number of molecules in dataset: 11044**


## Carboxylic acids


|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.85   | 0.81 |  0.88 | 
| Dissociation energy (electrofuge)  | 0.75	  | 0.75 | 	0.83 | 
| Electroaccepting power(w+)         | 0.63	  | 0.65 | 	0.66 | 
| Electrodonating power (w-)         | 0.79	  | 0.76 | 	0.84 | 
| Electronegativity (chi=-mu)        | 0.81	  | 0.88 | 	0.88 | 
| Electronic chemical potential (mu) | 0.81	  | 0.76 | 	0.88 | 
| Electronic chemical potential (mu+)| 0.72	  | 0.82 | 	0.77 | 
| Electronic chemical potential (mu-)| 0.86	  | 0.85 | 	0.91 | 
| Electrophilicity index (w=omega)   | 0.72	  | 0.80 | 	0.79 | 
| Global Dual Descriptor Deltaf+     | 0.59	  | 0.39 | 	0.61 | 
| Global Dual Descriptor Deltaf-     | 0.59	  | 0.66 | 	0.61 | 
| Hardness (eta)                     | 0.80	  | 0.57 | 	0.83 | 
| Hyperhardness (gamma)              | 0.67	  | 0.71 | 	0.69 | 
| Net Electrophilicity               | 0.72	  | 0.79 | 	0.81 | 
| Softness (S)                       | 0.41	  | 0.60 | 	0.43 | 

