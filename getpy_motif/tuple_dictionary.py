import numpy as np

class VectorLookupDict:

    def __init__(self) -> None:
        self.data = {}

    def __getitem__(self, np_array):
        print(np_array)
        if type(np_array).__module__ == np.__name__: return self.data[np_array.tobytes()]
        return self.data.__getitem__(np_array)
    
    def __setitem__(self, key, value):
        self.data[key.tobytes()] = value
        