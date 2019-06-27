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
        - [1000]
    bias_init_consts:
        - 0.8
    dropout:
        - 0.75
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
    layer_sizes: [1000]
    dropout: 0.75
    bias_init_consts: 0.8
""", Loader=yaml.FullLoader)
