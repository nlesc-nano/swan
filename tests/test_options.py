from swan.models.input_validation import Options


data = {'a': 3, 'c': {'d': 42}}


def test_opts():
    """Test the Options class."""
    opts = Options(data)

    assert all((opts.a == data['a'], opts.c['d'] == data['c']['d']))

    # insertion
    opts.elem = 42

    assert opts['elem'] == 42
