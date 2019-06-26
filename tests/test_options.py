from swan.models.input_validation import Options


data = {'a': 3, 'c': {'d': 42}}


def test_opts():
    """
    Test the options class
    """
    opts = Options(data)

    assert all((opts.a == data['a'], opts.c['d'] == data['c']['d']))
