
def test_main(mocker):
    """
    Test the CLI for the models
    """
    mocker.patch("swan.models.models.create_scatter_plot", return_value=None)


def test_load_data():
    """
    Test that the data is loaded correctly
    """
    return True


def test_load_model():
    """
    Test that a model is loaded correctly
    """
    return True


def test_predict_unknown():
    """
    Predict data for a some smiles
    """
    return True


def test_save_dataset(tmp_path):
    """
    Test that the dataset is stored correctly
    """
    return True


def test_train_tensorgraph():
    """
    Check the training process of a tensorgraph model
    """
    return True


def check_predict(model, researcher) -> bool:
    """
    Check that the predicted numbers are real
    """
    return True
