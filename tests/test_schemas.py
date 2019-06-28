from swan.models.input_validation import (
    sklearn_schema, tensorgraph_schema)


def test_sklearn_schema():
    """
    Check input for sklearn models
    """
    d = {'name': "SKlearn", "model": "randomForest", "parameters": {"n_jobs": -1}}
    sklearn_schema.validate(d)


def test_tensorgraph_schema():
    """
    Check input for tensorgraph models
    """
    d = {'name': "TensorGraph", "model": "fcnet", "epochs": 100, "parameters": {"dropout": 0.75}}
    tensorgraph_schema.validate(d)
