import unittest
import numpy as np
import pandas as pd
import pytest

from src.metrics.ranking import RankingMetric
from src.metrics.ranking import MeasureAtK
from src.metrics.ranking import NDCG
from src.metrics.ranking import NCRR
from src.metrics.ranking import MRR
from src.metrics.ranking import Precision
from src.metrics.ranking import Recall
from src.metrics.ranking import FMeasure
from src.metrics.ranking import AUC
from src.metrics.ranking import MAP
from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL
from src.metrics.evaluate_ignite import indices_extract, indices_search

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

# class TestEval(unittest.TestCase):
def test_indices_extract(df_true, jobsid, userid):
    # jobsid = np.asarray([11,  6,  4, 10,  2,  5, 14,  3,  7, 12])
    ind_item, _ = indices_extract(df=df_true, x_list =jobsid, feature=DEFAULT_ITEM_COL)
    ind_user1, _ = indices_extract(df=df_true, x_list=userid, feature=DEFAULT_USER_COL)
    assert np.all(ind_item == np.asarray([2, 3, 5, 8]))
    assert np.all(ind_user1 == np.asarray([1]))


def test_indices_search(df_true, jobsid):
    test_items = df_true[DEFAULT_ITEM_COL].values
    test_rating = df_true[DEFAULT_RATING_COL].values
    ids = np.asarray([5, 2, 3, 8])
    # reco_jobsid = np.asarray([4, 10, 7])
    reco_indices, test_rating_sort = indices_search(test_items, jobsid, test_rating, ids)
    # assert np.all(reco_indices == np.asarray([0, 3, 2]))
    assert np.all(reco_indices == np.asarray([1, 0, 3, 2]))
    assert np.all(test_rating_sort == np.asarray([0,0,0,1]))

# if __name__ == "__main__":
#     unittest.main()
