#!/usr/bin/env python

import sys
import os
import getpy
import numpy as np

file = sys.argv[1]


motif_dict = getpy.Dict(np.dtype('uint64'), np.dtype('uint64'), 0)
motif_dict.load(file)


keys_name = file.replace(".dump", "_keys.dump")
values_name = file.replace(".dump", "_values.dump")

keys = motif_dict.keys()
values = motif_dict.values()

np.save(open(keys_name, "wb"), keys, allow_pickle=False)
np.save(open(values_name, "wb"), values, allow_pickle=False)