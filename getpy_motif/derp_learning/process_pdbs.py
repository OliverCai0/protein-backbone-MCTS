import sys
import os
import _pickle
import concurrent.futures
import threading
import random
import argparse

import tqdm
import pyrosetta

from derp_learning.pdbdata import pdbdata
from derp_learning.util import InProcessExecutor, fnames_from_arg_list, cpu_count

pyrosetta.init("-beta_nov16 -mute all")


def output_fname(fname, storage_dir):
    outfile = fname.lstrip("/") + ".pickle"
    outfile = os.path.join(storage_dir, outfile)
    posefile = outfile[:-7] + ".pose_pickle"
    return outfile, posefile


def process_pdb(fname, storage_dir="."):
    outfile, posefile = output_fname(fname, storage_dir)

    try:
        # if os.path.exists(posefile):
        # with open(posefile, 'rb') as inp:
        # pose = _pickle.load(inp)
        # else:
        pose = pyrosetta.pose_from_file(fname)
    except:
        # print("ERROR pyrosetta can't read", fname)
        return fname

    # try:
    dat = pdbdata(pose, fname)
    # except Exception as e:
    # print("ERROR can't process pose from", fname)
    # print("   ", e)
    # return fname

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "wb") as out:
        _pickle.dump(dat, out)
    # if not os.path.exists(posefile):
    # with open(posefile, "wb") as out:
    # _pickle.dump(pose, out)


def process_parallel(pdbs, max_workers):
    if max_workers < 0:
        max_workers = cpu_count()
    if max_workers == 0:
        exe = InProcessExecutor()
    else:
        exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    errors = list()
    with exe as pool:
        futures = list()
        for pdb in pdbs:
            futures.append(pool.submit(process_pdb, pdb))
        iter = concurrent.futures.as_completed(futures)
        iter = tqdm.tqdm(iter, "processing pdb files", total=len(futures))
        for f in iter:
            err = f.result()
            if err is not None:
                errors.append(err)
    return errors


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", default=-1, type=int)
    parser.add_argument("pdbs", nargs="*")
    args = parser.parse_args(sys.argv[1:])

    pdbs = fnames_from_arg_list(args.pdbs, [".pdb", ".pdb.gz"])
    random.shuffle(pdbs)
    print("pdbs", len(pdbs))
    errors = process_parallel(pdbs, max_workers=args.parallel)
    if errors:
        print("pdb files not processed:")
        for e in errors:
            print("   ", e)
    else:
        print("processed all files")


if __name__ == "__main__":
    main()
