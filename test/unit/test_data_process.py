import os
import sys
sys.path.append('../../../src')
from src.data_process.neg_sample import random_sample
from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal


@pytest.fixture
def all_data():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 1, 2, 2],
        DEFAULT_ITEM_COL: [3, 4, 5, 6, 7]
    })


@pytest.fixture
def test_data():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1, 2],
        DEFAULT_ITEM_COL: [3, 6]
    })


@pytest.fixture
def train_data():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 2],
        DEFAULT_ITEM_COL: [4, 5, 7]
    })


def test_random_sample(all_data, test_data, train_data):
    df_train_neg = random_sample(train_data, all_data, 1)
    for index, value in enumerate(df_train_neg.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]):
        bool_set = set(~value[1].isin(all_data[all_data[DEFAULT_USER_COL] == value[0]][DEFAULT_ITEM_COL]))
        assert len(bool_set) == 1, 'There are items be both negative and postive'

    # print('aa')
    # df_train_neg_gt = pd.DataFrame({
    #
    # })
