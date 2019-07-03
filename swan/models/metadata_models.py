import yaml

__all__ = ["data_hyperparam_search", "default_hyperparameters"]

data_hyperparam_search = yaml.load("""
randomforest:
    n_estimators:
        - 10
        - 100
    max_features:
        - auto
        - sqrt
        - log2
    n_jobs:
        - -1
svr:
    kernel:
        - linear
        - poly
        - rbf
        - sigmoid
kernelridge:
    kernel:
        - linear
        - poly
        - rbf
        - sigmoid
bagging:
    n_estimators:
        - 10
        - 100
    max_features:
        - 0.5
        - .75
        - 1.0
    n_jobs:
        - -1
multitaskregressor:
    layer_sizes:
        - [50]
        - [100]
        - [200]
        - [500]
        - [100]
    bias_init_consts:
        - 0.8
        - 0.9
    dropout:
        - 0.75
        - 0.5
        - 0.0
    weight_init_stddevs:
        - 0.02
        - 0.06
""", Loader=yaml.FullLoader)


default_hyperparameters = yaml.load("""
randomforest:
    n_estimators: 100
    max_features: auto
    n_jobs: -1
svr:
    kernel:
        linear
kernelridge:
    kernel:
         rbf
bagging:
    n_estimators: 100
    max_features: 0.5
    n_jobs: -1
multitaskregressor:
    layer_sizes: [80]
    dropout: 0.5
    bias_init_consts: 0.8
    weight_init_stddevs: 0.06
""", Loader=yaml.FullLoader)
