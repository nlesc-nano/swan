class Options(dict):
    """
    Extend the base class dictionary with a '.' notation.
    example:
    .. code-block:: python
       d = Options({'a': 1})
       d['a'] # 1
       d.a    # 1
    """

    def __init__(self, *args, **kwargs):
        """ Create a recursive Options object"""
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Options(v)

    def __getattr__(self, attr):
        """ Allow `obj.key` notation"""
        return self.get(attr)

    def __setattr__(self, key, value):
        """ Allow `obj.key = new_value` notation"""
        self.__setitem__(key, value)
