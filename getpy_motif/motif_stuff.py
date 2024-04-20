!cd /home/drhicks1

import sys

sys.path.append("derp_learning")

import pickle

import numpy as np

import getpy

from numba import njit

def pair2mask( a, b ):
    if not isinstance(a, np.ndarray) and not isinstance(a, list):
        a = np.array(list(a), np.uint64)
    elif not isinstance(a, np.ndarray):
        a = np.array(a, np.uint64)

    if not isinstance(b, np.ndarray) and not isinstance(b, list):
        b = np.array(list(b), np.uint64)
    elif not isinstance(b, np.ndarray):
        b = np.array(b, np.uint64)

    value = a*20+b

    # least_signicant_encoding
    mask = np.zeros((len(a), 7), np.uint64)

    bit = value % 63

    value //= 63
    byte = value

    bit_mask = np.left_shift( np.uint64(1), bit.astype(np.uint64), dtype=np.uint64 )

    pairs = np.zeros((len(a), 2), np.uint64)
    pairs[:,0] = np.arange(0, len(a), dtype=np.uint64).astype(np.uint64)

    pairs[:,1] = byte

    mask[tuple(pairs.T)] = bit_mask

    return mask


def pairs2mask( a, b , both_directions=False ):

    mask = np.array([0,0,0,0,0,0,0], dtype=np.uint64)
    for i, j in zip(a, b):
        forward_mask = pair2mask(np.array([i], dtype=np.uint64), np.array([j], dtype=np.uint64))
        mask = np.bitwise_or(forward_mask, mask)

        if both_directions:
            reverse_mask = pair2mask(np.array([j], dtype=np.uint64), np.array([i], dtype=np.uint64))
            mask = np.bitwise_or(reverse_mask, mask)

    return mask


def mask2matrix( masks ):

    out = np.zeros((len(masks), 20, 20), np.bool)

    count = 0
    for byte in range(7):
        bit_mask = np.uint64(1)
        for bit in range(63):

            large_ind = count // 20
            small_ind = count % 20

            out[:,large_ind,small_ind] = np.bitwise_and( masks[:,byte], bit_mask, dtype=np.uint64 ) > 0

            bit_mask <<= np.uint64(1)
            count += 1

            if ( count >= 400 ):
                break
        if ( count >= 400 ):
            break

    return out

@njit (fastmath = True)
def fill_masks(motif_masks, indexes, masks):
    for i in range(len(indexes)):
        index = indexes[i]
        motif_masks[index] = np.bitwise_or(motif_masks[index], masks[i])

db = pickle.load(open("/home/sheffler/debug/derp_learning/datafiles/pdb_res_pair_data.pickle", 'rb'))

res_j = db['p_resj'][db['p_etot'] < -1.5]

res_i = db['p_resi'][db['p_etot'] < -1.5]

aa_i = db['aaid'][db['p_resi']][db["p_etot"] < -1.5]

aa_j = db['aaid'][db['p_resj']][db["p_etot"] < -1.5]

xijbin = np.array(db["xijbin_1.0_15"][db["p_etot"] < -1.5])

xjibin = np.array(db["xjibin_1.0_15"][db["p_etot"] < -1.5])

masks = pair2mask(aa_i, aa_j)

motif_dict = getpy.Dict(np.dtype('uint64'), np.dtype('uint64'))

all_keys = np.unique(np.concatenate((xijbin, xjibin)))

all_indx = np.arange(1, len(all_keys)+1,dtype=np.uint64)

motif_dict[all_keys] = all_indx

motif_masks = np.zeros((len(all_keys), 7), dtype=np.uint64)

xijidx = motif_dict[xijbin]
xjiidx = motif_dict[xjibin]

fill_masks(motif_masks, xijidx, masks)
fill_masks(motif_masks, xjiidx, masks)


# test #
for matrix in mask2matrix(motif_masks[motif_dict[all_keys[:100]]]):
    print()
    print(np.array(np.where(matrix)).T)
# test #

keys = all_keys[:100]
key_hits = motif_dict.__contains__(keys)
value_hits = motif_masks[motif_dict[keys[key_hits]]]

filter_mask = pairs2mask([0, 9, 17], [9, 9, 9], both_directions=True)

np.sum(np.any(np.bitwise_and(value_hits, filter_mask), axis=1))

"""
motif_dict.dump("/home/drhicks1/motif_dict.dump")
motif_dict = getpy.Dict(np.dtype('uint64'), np.dtype('uint64'))
motif_dict.load("/home/drhicks1/motif_dict.dump")
"""

"""
np.save(open("/home/drhicks1/motif_array.dump", 'wb'), motif_masks, allow_pickle=False)
motif_masks = np.load(open("/home/drhicks1/motif_array.dump", 'rb'), allow_pickle=False)
"""

"""
[[12 19]]

[[19 14]]

[[12 17]]

[[9 9]]

[[19 18]]

[[ 7  4]
 [13  9]]

[[18 12]]

[[ 9 18]]

...
"""


""" 
to use

A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y
0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19

create bitwise mask for aa_i by aa_j of interest (ie AxL, LxL, VxL) as rxr_of_interest
rxr_of_interest = pair2mask([0, 9, 17], [9, 9, 9])

array_of_xforms = xforms for each residue by residue of interest in pose/protein/npose

number_of_hits = sum(np.any(rxr_of_interest[motif_dict[array_of_xforms]]))

"""


