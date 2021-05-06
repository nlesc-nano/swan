# Machine learning model to predict CDFT properties

The following table shows the **mean rvalue** resulting from the linear regression
between the predicted as function of the ground true labels. Therefore, the closer
these values are to 1 the better the better the model is to predict the properties *with
the available amount of training data*.

The mean values where computing with the results of **3** training/validation datasets picked randomly
in a 0.8/0.2 proportion, respectively.

The hyperparameters use to perform the training can be found at [model's hyperparameters](Training_hyperparameters.md).


## Amines

**Number of molecules in dataset: 3109**

|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.73
| Dissociation energy (electrofuge)  | 0.39
| Electroaccepting power(w+)         | 0.42
| Electrodonating power (w-)         | 0.52
| Electronegativity (chi=-mu)        | 0.54
| Electronic chemical potential (mu) | 0.54
| Electronic chemical potential (mu+)| 0.38
| Electronic chemical potential (mu-)| 0.78
| Electrophilicity index (w=omega)   | 0.51
| Global Dual Descriptor Deltaf+     | 0.33
| Global Dual Descriptor Deltaf-     | 0.35
| Hardness (eta)                     | 0.53
| Hyperhardness (gamma)              | 0.39
| Net Electrophilicity               | 0.47
| Softness (S)                       | 0.42



**Number of molecules in dataset: 11044**


## Carboxylic acids


|          Property name             | FingerprintFullyConnected | MPNN | SE3Transformer|
|:----------------------------------:|:-------------------------:|:----:|:-------------:|
| Dissocation energy (nucleofuge)    | 0.85
| Dissociation energy (electrofuge)  | 0.75	 
| Electroaccepting power(w+)         | 0.63	 
| Electrodonating power (w-)         | 0.79	 
| Electronegativity (chi=-mu)        | 0.81	 
| Electronic chemical potential (mu) | 0.81	 
| Electronic chemical potential (mu+)| 0.72	 
| Electronic chemical potential (mu-)| 0.86	 
| Electrophilicity index (w=omega)   | 0.72	 
| Global Dual Descriptor Deltaf+     | 0.59	 
| Global Dual Descriptor Deltaf-     | 0.59	 
| Hardness (eta)                     | 0.80	 
| Hyperhardness (gamma)              | 0.67	 
| Net Electrophilicity               | 0.72	 
| Softness (S)                       | 0.41	
