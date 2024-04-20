import os
import numpy as np

datadir = os.path.join(os.path.dirname(__file__), "data")


def remove_redundant_pdbs(pdbs, sequence_identity=30):
    assert sequence_identity in (30, 40, 50, 70, 90, 95, 100)
    listfile = "pdbids_20190403_si%i.txt" % sequence_identity
    with open(os.path.join(datadir, listfile)) as inp:
        goodids = set(l.strip() for l in inp.readlines())
        assert all(len(g) == 4 for g in goodids)
    return np.array([i for i, p in enumerate(pdbs) if p[:4].upper() in goodids])
