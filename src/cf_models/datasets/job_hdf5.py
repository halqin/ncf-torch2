import os
import time
import logging
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from utils.constants import DEFAULT_USER_COL,DEFAULT_ITEM_COL,DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL


def _read_dataframe(filename, feather, logger):
    """Reads the original dataset TSV as a pandas dataframe"""
    # get a model based off the input params
    start = time.time()
    logger.debug("reading data from %s", filename)
    if feather:
        data = pd.read_feather(
            filename, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        )
    else:
        data = pd.read_csv(
            filename, usecols=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        )
    # map each artist and user to a unique numeric value
    data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].astype(str)
    data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].astype(str)
    data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].astype("category")
    data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].astype("category")
    # data[DEFAULT_RATING_COL] = data[DEFAULT_RATING_COL].astype("category")
    # store as a CSR matrix
    logger.debug("read data file in %s", time.time() - start)

    return data


def hdf5_from_dataframe(data, outputfilename):
    # create a sparse matrix of all the users/plays
    plays = coo_matrix(
        (
            data[DEFAULT_RATING_COL].astype(np.float32),
            (data[DEFAULT_USER_COL].cat.codes.copy(),
             data[DEFAULT_ITEM_COL].cat.codes.copy()),
        )
    ).tocsr()

    with h5py.File(outputfilename, "w") as f:
        g = f.create_group("job_user_apply")
        g.create_dataset("data", data=plays.data)
        g.create_dataset("indptr", data=plays.indptr)
        g.create_dataset("indices", data=plays.indices)

        dt = h5py.special_dtype(vlen=str)
        user = list(data[DEFAULT_USER_COL].cat.categories)
        dset = f.create_dataset(DEFAULT_USER_COL, (len(user),), dtype=dt)
        dset[:] = user

        jobs = list(data[DEFAULT_ITEM_COL].cat.categories)
        dset = f.create_dataset(DEFAULT_ITEM_COL, (len(jobs),), dtype=dt)
        dset[:] = jobs


def generate_dataset(data_path, data_name, logger):
    """Generates a hdf5 lastfm datasetfile from the raw datafiles:
    """
    indata = os.path.join(data_path, data_name)
    outdata = os.path.join(data_path, data_name+'.hdf5')
    feather_flag = not data_name.endswith('csv')
    data = _read_dataframe(filename=indata, feather=feather_flag, logger=logger)
    hdf5_from_dataframe(data, outputfilename=outdata)
    print('Finish generating hdf5')


def get_career(filename):
    """Returns the lastfm360k dataset, downloading locally if necessary.
    Returns a tuple of (artistids, userids, plays) where plays is a CSR matrix"""

    # filename = os.path.join(_download.LOCAL_CACHE_DIR, "lastfm_360k.hdf5")
    # if not os.path.isfile(filename):
    #     log.info("Downloading dataset to '%s'", filename)
    #     _download.download_file(URL, filename)
    # else:
    #     log.info("Using cached dataset at '%s'", filename)

    with h5py.File(filename, "r") as f:
        m = f.get("job_user_apply")
        plays = csr_matrix((m.get("data"), m.get("indices"), m.get("indptr")))
        return np.array(f[DEFAULT_ITEM_COL].asstr()[:]), np.array(f[DEFAULT_USER_COL].asstr()[:]), plays


def main():
    # job_info = pd.read_feather('../../data/clean/sub/jobs_sub')
    generate_dataset(data_path='../../data/clean',
                     data_name='toy_abc.csv',
                     logger=logging.getLogger("implicit")
                     )
    # jobs, users, job_user_apply = get_career('../../data/clean/sub/app.hdf5')


if __name__ == "__main__":
    main()



