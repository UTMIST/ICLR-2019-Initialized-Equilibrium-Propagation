def calc_relative_error(a, b):
    """
    a and b two scalars
    >>> calc_relative_error(5, 6)
    0.09090909090909091
    """
    return abs(a-b) / (abs(a) + abs(b))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
