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
        DEFAULT_RATING_COL: [1, 0, 0, 0, 0],
        'test_rating2': [0, 0, 1, 0, 0],
        'test_rating3': [0, 1, 0, 1, 1],
        'test_rating4': [0, 0, 0, 0, 0],
        'test_rating5': [1, 1, 1, 1, 1]
    })

# @pytest.fixture




@pytest.fixture
def userid():
    return np.asarray([2,1,3])

@pytest.fixture
def jobsid():
    return np.asarray([11,  6,  4, 10,  2,  5, 14,  3,  7, 12, 20])

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
    assert np.all(ind_item == np.asarray([2, 3, 5, 8, 10]))
    assert np.all(ind_user1 == np.asarray([1]))


def test_indices_search(df_true, jobsid):
    test_items = df_true[DEFAULT_ITEM_COL].values
    test_rating = df_true[DEFAULT_RATING_COL].values
    test_rating2 = df_true['test_rating3'].values
    ids = np.asarray([5, 2, 3, 8, 10]) # recommend itemid: 5, 4, 10, 7, 20
    ids2 = np.asarray([5, 2, 3, 8])
    score = np.asarray([.8, .7, .6, .5, .4])

    test_rating_sort,reco_indices, scores = indices_search(test_items, jobsid, test_rating, ids, score)
    test_rating_sort2,reco_indices2, scores2 = indices_search(test_items, jobsid, test_rating2, ids, score)
    test_rating_sort3, reco_indices3, scores3 = indices_search(test_items, jobsid, test_rating, ids2, score)

    assert np.all(reco_indices == np.asarray([1, 0, 3, 2, 4]))
    assert np.all(test_rating_sort == np.asarray([0,0,0,1,0]))
    assert np.all(scores==np.array([.7, .8, .5, .6, .4]))

    assert np.all(reco_indices2 == np.asarray([1, 0, 3, 2, 4]))
    assert np.all(test_rating_sort2 == np.asarray([0,1,1,0,1]))
    assert np.all(scores2==np.array([.7, .8, .5, .6, .4]))

    assert np.all(reco_indices3 == np.asarray([1, 0, 3, 2]))
    assert np.all(test_rating_sort3 == np.asarray([0,0,0,1]))
    assert np.all(scores3==np.array([.7, .8, .5, .6]))



def test_model_infer2(df_true, jobsid):
    model = BayesianPersonalizedRanking()
    model.recommend=MagicMock(return_value= (np.array([[5, 2, 3, 8, 10]]), np.array([.8, .7, .6, .5,.4])))
    gt_pos_bpr, reco_ind_bpr, scores_bpr = model_infer2(df_true=df_true, jobsid=jobsid, usersid=jobsid, 
                                 model=model, u_i_matrix=jobsid, n=1)
    assert np.all(gt_pos_bpr == np.asarray([0,0,0,1, 0]))
    # assert np.all(reco_ind_bpr== np.asarray([1,0,3,2]))
    np.testing.assert_array_equal(reco_ind_bpr, np.asarray([1,0,3,2,4]))
    np.testing.assert_array_equal(scores_bpr, np.asarray([.7,.8,.5,.6,.4]))

# if __name__ == "__main__":
#     unittest.main()
