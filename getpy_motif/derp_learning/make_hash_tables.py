import os
import _pickle
import time
import argparse
import sys
import random
from derp_learning.util import get_process_executor

import numpy as np
import xarray as xr
import numba as nb
from derp_learning.data_types import ResPairData
from derp_learning.khash import KHashi8i8


def load_which_takes_forever():
    print("loading huge dataset")
    t = time.perf_counter()
    with open("datafiles/pdb_res_pair_data_reduced_si90.pickle", "rb") as inp:
        rp = _pickle.load(inp)
    rp.sanity_check()
    print("load huge dataset time", time.perf_counter() - t)
    return rp


if hasattr(os, "a_very_unique_name"):
    respairdat = os.a_very_unique_name
else:
    respairdat = load_which_takes_forever()
    os.a_very_unique_name = respairdat


@nb.njit(nogil=True, fastmath=True)
def identical_ranges(x):
    assert x.ndim == 1
    prev = -1
    vals = np.zeros(len(x), dtype=np.int64) - 1
    ranges = np.zeros((len(x), 2), dtype=np.int32) - 1
    n = 0
    for i in range(len(x)):
        if x[i] != prev:
            ranges[n][0] = i
            vals[n] = x[i]
            if n > 0:
                ranges[n - 1][1] = i
            n += 1
            prev = x[i]
    ranges[n - 1, 1] = len(x)
    assert n == len(set(x))
    return vals[:n].copy(), ranges[:n].copy()


def make_xarray_binner(rp, bintype):
    binid = rp.data[bintype].values
    bin_srange2pairid = np.argsort(binid)
    sorted_binids = binid[bin_srange2pairid]
    binkeys, binvals = identical_ranges(sorted_binids)
    bin_srange = xr.DataArray(binvals, [("binidx", binkeys), ("bound", ["lb", "ub"])])
    return bin_srange, bin_srange2pairid


def make_numba_binner(rp, bintype):
    binid = rp.data[bintype].values
    bin_srange2pairid = np.argsort(binid)
    sorted_binids = binid[bin_srange2pairid]
    binkeys, ranges = identical_ranges(sorted_binids)
    binvals = ranges[:, 1].astype("i8") + np.left_shift(ranges[:, 0].astype("i8"), 32)
    bin_srange = KHashi8i8()
    bin_srange.update2(binkeys, binvals)

    assert np.all(ranges >= 0)
    assert np.all(binvals >= 0)
    assert bin_srange.size() == len(binkeys)
    assert np.all(bin_srange.array_get(binkeys) == binvals)
    assert np.all(np.right_shift(binvals, 32) == ranges[:, 0])
    assert np.all(binvals.astype("i4") == ranges[:, 1])

    return bin_srange, bin_srange2pairid


def get_bin_info(rp, bintype):
    bin_srange, bin_srange2pairid = make_xarray_binner(rp, bintype)

    ninbin = bin_srange.sel(bound="ub") - bin_srange.sel(bound="lb")
    assert np.all(ninbin > 0)
    # print("frac not one", np.sum(ninbin != 1) / len(binvals))
    # h = np.histogram(ninbin, bins="sturges")
    ssep = rp.p_resj - rp.p_resi  # offsets cancel
    unique, counts = np.unique(ninbin, return_counts=True)
    print("bin_count: nbins_wtih_count", dict(zip(unique, counts)))

    topbins = np.argsort(-ninbin).values[:10]
    for ibin in topbins:
        sep = ssep[rp[bintype] == bin_srange.binidx[ibin]].values
        print("xbin size", ninbin[ibin].values, "seqsep", np.mean(sep))

    # sanity check, all members of bin have bind == binkey
    binid = rp[bintype]
    sel = bin_srange.sel(binidx=binid[bin_srange2pairid])
    for pairid in np.random.choice(len(rp.pairid), 100):
        lb, ub = bin_srange.sel(binidx=binid[pairid]).values
        assert np.all(rp.data[bintype][bin_srange2pairid[lb:ub]] == binid[pairid])


@nb.njit(nogil=True, fastmath=True)
def pairs_to_seqprof(
    vbinidx,
    v_p_resi,
    v_p_resj,
    tbinidx,
    p_resi,
    p_resj,
    p_etot,
    p_dist,
    aaid,
    bin_trange,
    bin_trange2pairid,
    prof,
):
    nmissing, nfound = 0, 0
    for ivpair in range(len(v_p_resi)):
        binid = vbinidx[ivpair]
        trange = bin_trange.get(binid)
        if trange < 0:
            nmissing += 1
            continue
        nfound += 1
        lb = np.right_shift(trange, 32)
        ub = np.int32(trange)
        v_resi = v_p_resi[ivpair]
        v_resj = v_p_resj[ivpair]
        for itrange in range(lb, ub):
            itpair = bin_trange2pairid[itrange]
            assert tbinidx[itpair] == binid
            aai = aaid[p_resi[itpair]]
            aaj = aaid[p_resj[itpair]]
            prof[v_resi, aai] += 1.0
            prof[v_resj, aaj] += 1.0

    return nmissing, nfound


def make_seq_prof_xbins(t, v, bintypes, pc, min_ssep):
    prof = np.zeros((len(v.resid), 20), np.float32)

    for bintype in bintypes:
        bin_trange, bin_trange2pairid = make_numba_binner(t, bintype)
        nmissing, nfound = pairs_to_seqprof(
            vbinidx=v[bintype].values,
            v_p_resi=v.p_resi.values,
            v_p_resj=v.p_resj.values,
            tbinidx=t[bintype].values,
            p_resi=t.p_resi.values,
            p_resj=t.p_resj.values,
            p_etot=t.p_etot.values,
            p_dist=t.p_dist.values,
            aaid=t.aaid.values,
            bin_trange=bin_trange,
            bin_trange2pairid=bin_trange2pairid,
            prof=prof,
        )

    v_aafreq = np.zeros(20, np.float32)
    np.add.at(v_aafreq, v.aaid.values, 1)
    h_aafreq = np.sum(prof, axis=0)
    v_aafreq /= np.sum(v_aafreq)
    h_aafreq /= np.sum(h_aafreq)
    bias = h_aafreq / v_aafreq

    frac_found = nfound / len(v.p_resi)
    # print(nmissing, nfound, v.aaid[0].values, prof[0])

    aahat1 = np.argmax(prof, axis=1)
    frac1 = np.sum(aahat1 == v.aaid.values)
    sr1 = frac1 / len(aahat1)
    # print("sequence recovery", frac1 / len(aahat1))

    prof2 = prof / bias
    aahat2 = np.argmax(prof2, axis=1)
    frac2 = np.sum(aahat2 == v.aaid.values)
    sr2 = frac2 / len(aahat2)
    # print("unbiased recovery", frac2 / len(aahat2))

    prof3 = prof / np.sum(prof + 1, axis=1)[:, np.newaxis]
    aahat3 = np.argmax(prof3, axis=1)
    frac3 = np.sum(aahat3 == v.aaid.values)
    sr3 = frac3 / len(aahat3)
    # print("prior 1 recovery ", frac3 / len(aahat3))

    prof4 = prof / np.sum(prof + 1, axis=0)
    aahat4 = np.argmax(prof4, axis=1)
    frac4 = np.sum(aahat4 == v.aaid.values)
    sr4 = frac4 / len(aahat4)
    # print("norm recovery    ", frac4 / len(aahat4))

    srnb = [
        np.sum(aahat1[v.nnb10 >= t] == v.aaid[v.nnb10 >= t]) / np.sum(v.nnb10 >= t)
        for t in [14, 19, 24]
    ]

    pc = time.perf_counter() - pc
    print(
        f"frac found {frac_found:1.5f} seq recov",
        f"{sr1:1.5f} {sr2:1.5f} {sr3:1.5f} {sr4:1.5f} time {pc:8.3f} {len(t.pdb):6} {min_ssep:3}",
    )
    return [sr1, frac_found] + srnb


def run_seq_prof_test(N, ijob, min_ssep=0):

    np.random.seed(np.random.randint(2 ** 24) + ijob)

    # pc = time.perf_counter()
    if N >= len(respairdat.pdb):
        rp = respairdat
    else:
        rp = respairdat.subset_by_pdb(N, random=1, sanity_check=1)
    # print("time subset", time.perf_counter() - pc)

    # binners = {bintype: make_numba_binner(rp, bintype) for bintype in rp.xbin_types}
    # print(binners)
    # get_bin_info(rp, "xijbin_2.0_30")

    # pc = time.perf_counter()
    ssep = rp.p_resj - rp.p_resi
    rp2 = rp.subset_by_pair(ssep >= min_ssep, sanity_check=1)
    # print("time subset_pair", time.perf_counter() - pc, "del rp")

    # pc = time.perf_counter()
    train, valid = rp2.split_by_pdb(0.75, random=1, sanity_check=1)
    # print("time split", time.perf_counter() - pc, "del rp2")

    pc = time.perf_counter()
    bintype = ["xijbin_1.0_15", "xjibin_1.0_15"]
    # bintype = ["xijbin_1.0_15"]
    # bintype = ["xjibin_1.0_15"]

    out = make_seq_prof_xbins(train, valid, bintype, pc, min_ssep)

    return out


def find_subset_error(i):
    try:
        print(i)
        seed = np.random.randint(2 ** 32)
        np.random.seed(seed)
        stage = "subset_by_pdb"
        rp = respairdat.subset_by_pdb(10, random=1, sanity_check=1)
        stage = "subset_by_pair"
        rp = rp.subset_by_pair(rp.p_resj - rp.p_resi >= 9, sanity_check=1)
        stage = "split"
        train, valid = rp.split_by_pdb(0.5, random=1, sanity_check=1)
    except:
        print("FAIL", stage, "seed:", seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=-1, type=int)
    args = parser.parse_args(sys.argv[1:])

    # with get_process_executor(args.parallel) as pool:
    # pool.map(find_subset_error, range(1000000))
    # return

    # np.random.seed(3094865702)

    # respairdat.sanity_check()

    nsamp = 28
    # sizes = [len(respairdat.pdb)]
    sizes = [1000]
    seq_seps = [1, 4, 9, 16]

    srecov = list()
    ffound = list()
    srnb = list()
    rtime = list()
    rsize = list()
    rssep = list()

    t = time.perf_counter()
    with get_process_executor(args.parallel) as pool:

        for N in sizes:
            for ssep in seq_seps:
                print("start", N, ssep)
                ttmp = time.perf_counter()
                rsize.append(N)
                rssep.append(ssep)
                futures = list()
                for i in range(nsamp):
                    futures.append(pool.submit(run_seq_prof_test, N, i, ssep))
                result = np.stack([f.result() for f in futures])
                srecov.append(np.mean(result[:, 0], axis=0))
                ffound.append(np.mean(result[:, 1], axis=0))
                srnb.append(np.mean(result[:, 2:], axis=0))
                rtime.append(time.perf_counter() - ttmp)

    for i, sr in enumerate(srecov):
        print(
            f"result ntot: {rsize[i]:6} ssep: {rssep[i]:3}",
            f"seqrecov: {sr:1.4f} ffound: {ffound[i]:1.4f} time: {rtime[i]:6.1f}",
            f"nb13 {srnb[i][0]:1.4f} nb18 {srnb[i][1]:1.4f} nb23 {srnb[i][2]:1.4f} ",
        )


if __name__ == "__main__":
    main()


# avg seq recov 0.17891613973143483 time 1372.4419116459903
# mean seq recov 100 0.16401754360238108
# mean seq recov 330 0.16608294672397578
# mean seq recov 1000 0.1753792602071716
# mean seq recov 2713 0.17891613973143483


# SI30 mean seq recov    100   1 0.14889   25.4
# SI30 mean seq recov   1000   1 0.15491  296.1
# SI30 mean seq recov  10000   1 0.15829 43035.0

# SI30 mean seq recov    100   1 0.15219   24.4
# SI30 mean seq recov    100   4 0.11992   18.2
# SI30 mean seq recov    100   8 0.10110   17.9

# result ntot:    100 ssep:   1 seqrecov: 0.1548 ffound: 0.3696 time:   29.2 nb13 0.1586 nb18 0.1692 nb23 0.1849
# result ntot:    100 ssep:   4 seqrecov: 0.1186 ffound: 0.1831 time:   19.7 nb13 0.1257 nb18 0.1396 nb23 0.1584
# result ntot:    100 ssep:   9 seqrecov: 0.0980 ffound: 0.0886 time:   19.5 nb13 0.1053 nb18 0.1210 nb23 0.1451
# result ntot:    100 ssep:  16 seqrecov: 0.0945 ffound: 0.0735 time:   19.4 nb13 0.1014 nb18 0.1170 nb23 0.1431
# result ntot:    330 ssep:   1 seqrecov: 0.1572 ffound: 0.4316 time:   34.2 nb13 0.1608 nb18 0.1719 nb23 0.1903
# result ntot:    330 ssep:   4 seqrecov: 0.1319 ffound: 0.2402 time:   18.2 nb13 0.1392 nb18 0.1531 nb23 0.1703
# result ntot:    330 ssep:   9 seqrecov: 0.1069 ffound: 0.1348 time:   16.9 nb13 0.1168 nb18 0.1335 nb23 0.1559
# result ntot:    330 ssep:  16 seqrecov: 0.1052 ffound: 0.1191 time:   16.9 nb13 0.1145 nb18 0.1311 nb23 0.1541
# result ntot:   1000 ssep:   1 seqrecov: 0.1566 ffound: 0.4995 time:  293.7 nb13 0.1607 nb18 0.1723 nb23 0.1899
# result ntot:   1000 ssep:   4 seqrecov: 0.1443 ffound: 0.3214 time:   50.6 nb13 0.1524 nb18 0.1667 nb23 0.1841
# result ntot:   1000 ssep:   9 seqrecov: 0.1211 ffound: 0.2091 time:   23.9 nb13 0.1324 nb18 0.1504 nb23 0.1723
# result ntot:   1000 ssep:  16 seqrecov: 0.1132 ffound: 0.1865 time:   23.5 nb13 0.1238 nb18 0.1410 nb23 0.1634
