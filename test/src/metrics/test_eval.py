from ast import GtE
import unittest
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
# from src.implicit_build import bpr, bpr, lmf
from implicit.bpr import BayesianPersonalizedRanking

from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL
from src.metrics.evaluate_ignite import indices_extract, indices_search, model_infer2

@pytest.fixture
def df_true():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1,1,1,1,1 ],
        DEFAULT_ITEM_COL: [10,5,4,7,20],
        DEFAULT_RATING_COL: [1, 0, 0, 0, 0]
    })

@pytest.fixture
def userid():
    return np.asarray([2,1,3])

@pytest.fixture
def jobsid():
    return np.asarray([11,  6,  4, 10,  2,  5, 14,  3,  7, 12])

# @pytest.fixture
# def df_score():
#     return np.asarray([.7, .4, .8, .6, .5])

# @pytest.fixture
# def pre_score():
#     return np.asarray([.8, .7, .6, .5])


# class TestEval(unittest.TestCase):
def test_indices_extract(df_true, jobsid, userid):
    # jobsid = np.asarray([11,  6,  4, 10,  2,  5, 14,  3,  7, 12])
    ind_item, _ = indices_extract(df=df_true, x_list =jobsid, feature=DEFAULT_ITEM_COL)
    ind_user1, _ = indices_extract(df=df_true, x_list=userid, feature=DEFAULT_USER_COL)
    assert np.all(ind_item == np.asarray([2, 3, 5, 8]))
    assert np.all(ind_user1 == np.asarray([1]))


def test_indices_search(df_true, jobsid, ):
    test_items = df_true[DEFAULT_ITEM_COL].values
    test_rating = df_true[DEFAULT_RATING_COL].values
    ids = np.asarray([5, 2, 3, 8]) # recommend itemid: 5, 4, 10, 7
    # pred_score = np.asarray([.8, .7, .6, .5])
    test_rating_sort,reco_indices = indices_search(test_items, jobsid, test_rating, ids)
    assert np.all(reco_indices == np.asarray([1, 0, 3, 2]))
    assert np.all(test_rating_sort == np.asarray([0,0,1,0]))
    # assert np.all(pred_score[reco_indices]==np.array([.8, .7, .6, .5]))


def test_model_infer2(df_true, jobsid):
    model = BayesianPersonalizedRanking()
    model.recommend=MagicMock(return_value= (np.array([5, 2, 3, 8]), np.array([.8, .7, .6, .5])))
    gt_pos_bpr, reco_ind_bpr, scores_bpr = model_infer2(df_true=df_true, jobsid=jobsid, usersid=jobsid, 
                                 model=model, u_i_matrix=jobsid, n=1)
    assert np.all(gt_pos_bpr == np.asarray([0,0,1,0]))
    assert np.all(reco_ind_bpr== np.asarray([1,0,3,2]))
    assert np.all(scores_bpr == np.asarray([.8,.7,.6,.5]))
# if __name__ == "__main__":
#     unittest.main()
