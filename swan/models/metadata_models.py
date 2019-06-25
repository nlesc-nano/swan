import yaml

data_hyperparam_search = yaml.load("""
randomforest:
    n_estimators:
        - 10
        - 100
    max_features:
        - auto
        - sqrt
        - log2
""", Loader=yaml.FullLoader)
