import os
import sys
sys.path.append('../../src')
from hydra import initialize, compose
import pathlib
import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm

# from datasets.jobdataset import generate_dataset, _hfd5_from_dataframe
from datasets.job_hdf5 import hdf5_from_dataframe, get_career
# import data_process.neg_sample as ng_sample
from src.utils.constants import DEFAULT_USER_COL,DEFAULT_ITEM_COL,DEFAULT_RATING_COL, DEFAULT_PREDICTION_COL
# from implicit_eval import microsoft_eval,model_infer_df
# from implicit.als import AlternatingLeastSquares
# from implicit.bpr import BayesianPersonalizedRanking
# from implicit.lmf import LogisticMatrixFactorization
from implicit_build import bpr
from src.metrics import ranking
from src.metrics.evaluate_ignite import model_infer2

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config", overrides=[])

jobsid, usersid, user_job_app = get_career(pathlib.Path(cfg.path.root, cfg.file.hdf5))
model_path = "./models"
def read_train_gd_csv(data_testgd_path, usecols):
    test_gddf = pd.read_csv(data_testgd_path, usecols=usecols)
#     test_gddf[DEFAULT_USER_COL] = test_gddf[DEFAULT_USER_COL].astype('str')
#     test_gddf[DEFAULT_ITEM_COL] = test_gddf[DEFAULT_ITEM_COL].astype('str')
    return test_gddf
df_test_ori = pd.read_feather(pathlib.Path(cfg.path.root, cfg.file.test))


df_train =read_train_gd_csv('../../data/jobs/leave_one_train_neg.csv', usecols=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL])

df_test =read_train_gd_csv('../../data/jobs/leave_one_test.csv', usecols=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL])
with open('./models/model_bpr.sav','rb') as pickle_in:
    model = pickle.load(pickle_in)
apps_true = df_test_ori[df_test_ori['userid'].isin([1472090])]
gt_pos, reco_ind, scores = model_infer2(df_true=apps_true, jobsid=jobsid, usersid=usersid,
                                 model=model, u_i_matrix=user_job_app, n=cfg.params.neg_test+1)
p = ranking.Precision(k=20)
p.compute(gt_pos=gt_pos, pd_rank=reco_ind)