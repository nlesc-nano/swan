class Options(dict):
    """
    Extend the base class dictionary with a '.' notation.
    example:
    .. code-block:: python
       d = Options({'a': 1})
       d['a'] # 1
       d.a    # 1
    """

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __deepcopy__(self, _):
        return Options(self.copy())
