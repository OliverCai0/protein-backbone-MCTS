import sys
import numpy as np
import itertools
import xbin


def vectorized_set(d, keys, values):
    assert len(keys) == len(values)
    for k,v in zip(keys, values): d[k] = v

def vectorized_get(d, keys, default_value=0):
    return np.array([
        d[k] if k in d else default_value for k in keys
    ])

motif_dict = {}
# motif_dict.load("/home/bcov/from/derrick/getpy_motif/motif_dict_1.0_15_-1.0_H-H.dump")
keys = np.load(open("fmh/motif_dict_1.0_15_-1.0_H-H_keys.dump", 'rb'), allow_pickle=False)
values = np.load(open("fmh/motif_dict_1.0_15_-1.0_H-H_values.dump", 'rb'), allow_pickle=False)
vectorized_set(motif_dict, keys, values)
del keys
del values

motif_masks = np.load(open("fmh/motif_array_1.0_15_-1.0_H-H.dump", 'rb'), allow_pickle=False)

binner = xbin.XformBinner(1.0, 15, 512)

def pair2mask( a, b ):

    a = a.astype(np.uint64)
    b = b.astype(np.uint64)

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

def fill_masks(motif_masks, indexes, masks):
    for i in range(len(indexes)):
        index = indexes[i]
        motif_masks[index] = np.bitwise_or(motif_masks[index], masks[i])


aa = "ACDEFGHIKLMNPQRSTVWY"
aa2num = {}
for i, a in enumerate(aa):
    aa2num[a] = i


# takes 2 strings
def get_search_mask(these, by_these):
    num_these = [aa2num[x] for x in these]
    num_by_these = [aa2num[x] for x in by_these]
    combos = list(itertools.product(num_these, num_by_these))

    masks = pairs2mask(*list(zip(*combos)))

    return np.bitwise_or.reduce(masks, axis=-2)


def get_raw_hits(xforms):

    keys = binner.get_bin_index(xforms.reshape(-1, 4, 4)).astype(np.uint64)

    # print(keys)
    # print("Vectorized Gets")
    # print(vectorized_get(motif_dict, keys))
    raw_hits = motif_masks[
        # motif_dict[keys]
        vectorized_get(motif_dict, keys).astype(int)
        ]
    
    return raw_hits

def get_masked_hits(xforms, search_mask):
    raw_hits = get_raw_hits(xforms)

    return np.any( np.bitwise_and(raw_hits, search_mask), axis=-1 )


_atom_record_format = (
    "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}{seg:-4d}{elem:2s}\n"
)

def format_atom(
        atomi=0,
        atomn='ATOM',
        idx=' ',
        resn='RES',
        chain='A',
        resi=0,
        insert=' ',
        x=0,
        y=0,
        z=0,
        occ=1,
        b=0,
        seg=1,
        elem=''
):
    return _atom_record_format.format(**locals())

def dump_pts(pts, name):
    with open(name, "w") as f:
        for ivert, vert in enumerate(pts):
            f.write(format_atom(ivert%100000, resi=ivert%10000, x=vert[0], y=vert[1], z=vert[2]))

def dump_n_x_vs_x_hashes(n, x, by_x, name, random_size=0):

    reverse = {}
    keys = np.array(list(motif_dict))
    # values = motif_dict[keys]
    values = vectorized_get(motif_dict, keys)
    # reverse[values] = keys
    vectorized_set(reverse, values, keys)

    search_mask = get_search_mask(x, by_x)

    indices = np.where(np.any(np.bitwise_and(motif_masks, search_mask), axis=-1))[0]

    xbin_hashes = reverse[indices.astype(np.uint64)]

    assert(not np.any(xbin_hashes == 0))

    xforms = binner.get_bin_center(xbin_hashes)
    xforms[:,:3,3] += (np.random.random((len(xforms), 3)) - 0.5) * random_size

    pts = np.r_[ np.array([[0, 0, 0]]), xforms[:n, :3, 3]]

    dump_pts(pts, name)





