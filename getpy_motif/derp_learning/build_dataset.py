import os
import sys
import random
import _pickle
import argparse
from time import perf_counter

from tqdm import tqdm
import numpy as np
import xarray as xr

import xbin
from derp_learning.data_types import ResPairData
from derp_learning.util import fnames_from_arg_list, get_process_executor


def load(fname):
    with open(fname, "rb") as f:
        return _pickle.load(f)


def process_pdb_data(fnames, parallel):
    print(f"process_pdb_data for {len(fnames):,} files")
    if len(fnames) == 0:
        print("no files specified")
        sys.exit()

    raw = [load(f) for f in tqdm(fnames, "reading data files")]
    sanity_check_raw(raw)

    print("make per-structure data")
    pdb = [x["fname"] for x in raw]
    chains = [x["chains"] for x in raw]
    com = np.stack([x["coords"]["com"] for x in raw])
    rg = np.array([x["coords"]["rg"] for x in raw], np.float32)
    nres = np.array([len(x["coords"]["cb"]) for x in raw])
    pdbdata = dict(pdb=pdb, nres=nres, chains=chains, com=com, rg=rg)
    pdbenergies = {
        k: np.array([v["energies"][k] for v in raw], "f4") for k in raw[0]["energies"]
    }
    pdbdata = {**pdbdata, **pdbenergies}

    print("make per-residue data")
    n = np.concatenate([x["coords"]["ncac"][:, 0] for x in raw])
    ca = np.concatenate([x["coords"]["ncac"][:, 1] for x in raw])
    c = np.concatenate([x["coords"]["ncac"][:, 2] for x in raw])
    cb = np.concatenate([x["coords"]["cb"] for x in raw])
    o = np.concatenate([x["coords"]["o"] for x in raw])
    stubs = np.concatenate([x["coords"]["stubs"] for x in raw])
    coords = dict(n=n, ca=ca, c=c, o=o, cb=cb, stubs=stubs)

    resdata = dict()
    resdata["pdbno"] = [np.repeat(i, len(x["coords"]["cb"])) for i, x in enumerate(raw)]
    resdata["pdbno"] = np.concatenate(resdata["pdbno"]).astype("i4")
    resdata["resno"] = [np.arange(len(x["coords"]["cb"])) for x in raw]
    resdata["resno"] = np.concatenate(resdata["resno"]).astype("i4")
    for name in raw[0]["resdata"].keys():
        if type(raw[0]["resdata"][name]) is str:
            dat = np.concatenate([list(x["resdata"][name]) for x in raw])
        else:
            dat = np.concatenate([x["resdata"][name] for x in raw])
        resdata[name] = dat

    pdb_res_offsets = np.cumsum([len(x["resdata"]["phi"]) for x in raw])
    pdb_res_offsets = np.concatenate([[0], pdb_res_offsets])
    sanity_check_res_data(**vars())

    print("compute neighbor stats")
    nbdists = [6, 8, 10, 12, 14]
    compute_neighbor_stats(**vars())

    pairdata = dict()
    for name in tqdm(raw[0]["pairdata"], "make residue pair data"):
        pairdata[name] = np.concatenate([x["pairdata"][name] for x in raw])
    pairdata["p_resi"] = pairdata["p_resi"].astype("i4")
    pairdata["p_resj"] = pairdata["p_resj"].astype("i4")

    pdb_pair_offsets = np.cumsum([len(x["pairdata"]["p_dist"]) for x in raw])
    pdb_pair_offsets = np.concatenate([[0], pdb_pair_offsets])
    tmp = [np.repeat(i, len(x["pairdata"]["p_dist"])) for i, x in enumerate(raw)]
    pairdata["pdbno"] = np.concatenate(tmp).astype("i4")
    sanity_check_pair_res_relation(**vars())

    tot_nres = len(resdata["phi"])
    tot_pairs = len(pairdata["p_dist"])

    print("sanity check pair distances vs res-res distances")
    pair_resi_idx = pdb_res_offsets[pairdata["pdbno"]] + pairdata["p_resi"]
    pair_resj_idx = pdb_res_offsets[pairdata["pdbno"]] + pairdata["p_resj"]
    cbi = cb[pair_resi_idx]
    cbj = cb[pair_resj_idx]
    dhat = np.linalg.norm(cbi - cbj, axis=1)
    assert np.allclose(dhat, pairdata["p_dist"], atol=1e-3)

    print(f"returning: nstruct {len(pdb):,} nres {tot_nres:,} npairs {tot_pairs:,}")
    return dict(
        pdbdata=pdbdata,
        coords=coords,
        resdata=resdata,
        pairdata=pairdata,
        pdb_pair_offsets=pdb_pair_offsets,
        pdb_res_offsets=pdb_res_offsets,
        eweights=raw[0]["eweights"],
    )


def sanity_check_raw(raw):
    w0 = raw[0]["eweights"]
    print(w0)
    for r in raw:
        assert r["eweights"] == w0


def _compute_nnb(res_cb, nbdists):
    nnb = list()
    # 0.000001 hack to avoid some stupid sqrt err in np.linalg.norm
    diff = np.abs(res_cb[None, :] - res_cb[:, None]) + 0.000_001
    p_dist = np.linalg.norm(diff, axis=2)
    for nbdist in nbdists:
        lbl = "nnb" + str(nbdist)
        nnb.append(np.sum(p_dist <= nbdist, axis=1) - 1)
    return nnb


def compute_neighbor_stats(pdb, resdata, cb, nbdists, pdb_res_offsets, parallel, **kw):
    with get_process_executor(parallel) as pool:
        # for ipdb in tqdm(range(len(pdb)), "compute res num neighbor burial"):
        futures = list()
        for ipdb in range(len(pdb)):
            lb, ub = pdb_res_offsets[ipdb : ipdb + 2]
            res_cb = cb[lb:ub]
            futures.append(pool.submit(_compute_nnb, res_cb, nbdists))
            # foo
        result = np.concatenate([np.array(f.result()).T for f in futures])

    for i, nbdist in enumerate(nbdists):
        lbl = "nnb" + str(nbdist)
        resdata[lbl] = result[:, i]


def sanity_check_res_data(nres, pdb_res_offsets, resdata, **kw):
    for i, o in enumerate(tqdm(pdb_res_offsets[:-1], "sanity_check_res_data")):
        lb, ub = pdb_res_offsets[i : i + 2]
        assert np.all(resdata["pdbno"][lb:ub] == i)
        assert np.all(resdata["resno"][lb:ub] < nres[i])
        assert np.all(resdata["resno"][lb] == 0)


def sanity_check_pair_res_relation(pdb, nres, pdb_pair_offsets, pairdata, **kw):
    for i, o in enumerate(
        tqdm(pdb_pair_offsets[:-1], "sanity_check_pair_res_relation")
    ):

        try:
            if i < 100:
                pdbi = pairdata["pdbno"] == i
            for tmp in ("p_resi", "p_resj"):
                if i < 100:
                    residx = pairdata[tmp][pdbi]
                    assert max(residx) < nres[i]
                    assert min(residx) == 0 or tmp == "p_resj"
                lb, ub = pdb_pair_offsets[i : i + 2]
                residx = pairdata[tmp][lb:ub]
                assert max(residx) < nres[i]
                assert min(residx) == 0 or tmp == "p_resj"
        except AssertionError as e:
            print("Error on", pdb[i])
            raise e


def compute_pair_xform_bins(dat, parallel=-1):
    print(f"compute binnings for {len(dat['pairdata']['p_dist'])*2:,} pairs")

    pdb_res_offsets = dat["pdb_res_offsets"]
    pairdata = dat["pairdata"]
    stubs = dat["coords"]["stubs"]

    pair_resi_idx = pdb_res_offsets[pairdata["pdbno"]] + pairdata["p_resi"]
    pair_resj_idx = pdb_res_offsets[pairdata["pdbno"]] + pairdata["p_resj"]

    t = perf_counter()
    pair_stubi = stubs[pair_resi_idx]
    pair_stubj = stubs[pair_resj_idx]
    print("make xij")
    xij = np.linalg.inv(pair_stubi) @ pair_stubj
    print("make xji")
    xji = np.linalg.inv(pair_stubj) @ pair_stubi
    # assert np.allclose(xij @ xji, np.eye(4), atol=1e-3)  # floats kinda suck...
    print("make stubs & xij time", perf_counter() - t)

    print("compute bins")
    cart_resl = [0.5, 1.0, 2.0]
    ori_resl = [7.5, 15, 30]
    bin_parms = [(c, o) for c in cart_resl for o in ori_resl]
    dat["bin_params"] = bin_parms

    # t = perf_counter()
    indexers = [
        # xbin.gu_xbin_indexer(cart_resl=2.0, ori_resl=15.0) for c, o in bin_parms,
        xbin.XformBinner(c, o)
        for c, o in bin_parms
    ]
    # print("numba compile time", perf_counter() - t)

    nparts = 1024
    split_xs = [np.array_split(xij, nparts), np.array_split(xji, nparts)]

    t = perf_counter()
    dat["xbin_types"] = list()
    with get_process_executor(parallel) as pool:
        for i, indexer in enumerate(indexers):
            for j, split_x in enumerate(split_xs):
                t2 = perf_counter()
                futures = list()
                for part in split_x:
                    futures.append(pool.submit(indexer.get_bin_index, part))

                xhat = indexer.get_bin_center(futures[0].result())
                d = np.linalg.norm(xhat[:, :, 3] - split_x[0][:, :, 3], axis=1)
                # print("p_dist err", bin_parms[i], np.mean(d), np.min(d), np.max(d))
                assert np.max(d) < bin_parms[i][0] * 1.5

                result = np.concatenate([f.result() for f in futures])
                assert len(result) == len(pairdata["p_dist"])
                lbl = str(bin_parms[i][0]) + "_" + str(bin_parms[i][1])
                lbl = ["xijbin", "xjibin"][j] + "_" + lbl
                dat["xbin_types"].append(lbl)
                pairdata[lbl] = result
                # print("compute bins", i, j, lbl, "time", perf_counter() - t2)

    dat["xbin_swap_type"] = {
        x: x.replace("i", "?").replace("j", "i").replace("?", "j")
        for x in dat["xbin_types"]
    }

    print("binning time", perf_counter() - t)

    return dat


def print_summary(dat):
    print("  pdbdata")
    npdb = len(dat["pdbdata"]["nres"])
    nres = np.sum(dat["pdbdata"]["nres"])
    npair = len(dat["pairdata"]["p_dist"])
    for k, v in dat["pdbdata"].items():
        print("   ", k, f"{v.shape if hasattr(v, 'shape') else len(v)}")
        assert len(v) == npdb
    print("  coords")
    for k, v in dat["coords"].items():
        print("   ", k, f"{v.shape if hasattr(v, 'shape') else len(v)}")
        assert len(v) == nres
    print("  resdata")
    print("    offsets", dat["pdb_res_offsets"].shape)
    for k, v in dat["resdata"].items():
        print("   ", k, f"{v.shape if hasattr(v, 'shape') else len(v)}")
        assert len(v) == nres
    print("  pairdata")
    print("    offsets", dat["pdb_pair_offsets"].shape)
    for k, v in dat["pairdata"].items():
        print("   ", k, f"{v.shape if hasattr(v, 'shape') else len(v)}")
        assert len(v) == npair


def main():

    _bad_pickles = set(
        [
            "home/aivan/work/mds_training/PISCES181019/biounit/relaxed/b6/6b6u_0001.pdb.pickle",
            "home/aivan/work/mds_training/PISCES181019/biounit/relaxed/rf/2rfj_0001.pdb.pickle",
            "home/aivan/work/mds_training/PISCES181019/biounit/relaxed/rp/3rpe_0001.pdb.pickle",
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", default="pdb_res_pair_data.pickle")
    parser.add_argument("--parallel", default=-1, type=int)
    parser.add_argument("pdbs", nargs="*")
    args = parser.parse_args(sys.argv[1:])

    fnames = fnames_from_arg_list(args.pdbs, [".pickle"])
    fnames = [f for f in fnames if f not in _bad_pickles]

    # if os.path.exists("__HACK_TMP_FILE.pickle"):
    # print("reading hacky tmp file")
    # with open("__HACK_TMP_FILE.pickle", "rb") as inp:
    # t = perf_counter()
    # print("loading hack tmp pickle")
    # dat = _pickle.load(inp)
    # print("loading hack tmp pickle done", perf_counter() - t)
    # else:

    dat = process_pdb_data(fnames, args.parallel)

    dat = compute_pair_xform_bins(dat, args.parallel)

    #
    # with open("__HACK_TMP_FILE.pickle", "wb") as out:
    # _pickle.dump(dat, out)
    #

    print("summary of raw data")
    print_summary(dat)

    rp = ResPairData(dat)
    del dat

    print(rp)

    for t in rp.xbin_types:
        print("xbin_swap_type", t, "->", rp.xbin_swap_type[t])

    with open(args.outfile, "wb") as out:
        _pickle.dump(rp, out)


if __name__ == "__main__":
    main()
