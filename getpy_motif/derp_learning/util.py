import os
import glob
import multiprocessing
import concurrent.futures
import threading
import numba


jit = numba.njit(nogil=True, fastmath=True)


def cpu_count():
    try:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    except:
        return multiprocessing.cpu_count()


def get_process_executor(max_workers):
    if max_workers < 0:
        max_workers = cpu_count()
    if max_workers == 0:
        exe = InProcessExecutor()
    else:
        exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    return exe


class InProcessExecutor:
    def __init__(self, *args, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args, **kw):
        return NonFuture(fn, *args, **kw)


class NonFuture:
    def __init__(self, fn, *args, dummy=None, **kw):
        self.fn = fn
        self.dummy = not callable(fn) if dummy is None else dummy
        self.args = args
        self.kw = kw
        self._condition = threading.Condition()
        self._state = "FINISHED"
        self._waiters = []

    def result(self):
        if self.dummy:
            return self.fn
        return self.fn(*self.args, **self.kw)


def pdbs_from_file(fname, exts):
    # print("pdbs_from_file", fname)
    if fname.endswith(exts):
        return [fname]
    else:
        print("ERROR don't know now to handle file", fname)
        return []


def fnames_from_glob(pattern, exts):
    # print("fnames_from_glob", pattern)
    fnames = list()
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            fnames += fnames_from_directory(path, exts)
        else:
            fnames += pdbs_from_file(path, exts)
    return fnames


def fnames_from_directory(path, exts):
    # print("fnames_from_directory", path)
    fnames = list()
    for ex in exts:
        fnames += fnames_from_glob(path + "/*" + ex, exts)
    return fnames


def fnames_from_arg_list(args, exts):
    if type(exts) is not str:
        exts = tuple(exts)
    # print("fnames_from_arg_list", args)
    fnames = list()
    for arg in args:
        if os.path.isdir(arg):
            fnames += fnames_from_directory(arg, exts)
        elif arg.count("*") or arg.count("?"):
            fnames += fnames_from_glob(arg, exts)
        else:
            fnames += pdbs_from_file(arg, exts)
    return fnames
