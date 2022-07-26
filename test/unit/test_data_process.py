from src.data_process.neg_sample import random_sample
from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal


@pytest.fixture
def all_data():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 1, 2, 2, 3, 3, 3, 3],
        DEFAULT_ITEM_COL: [3, 4, 5, 6, 7, 10,11,12,13]
    })


@pytest.fixture
def test_data():
    df1 = pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 2],
        DEFAULT_ITEM_COL: [4, 5, 7]
    })

    df2 =  pd.DataFrame({
        DEFAULT_USER_COL: [1, 1],
        DEFAULT_ITEM_COL: [4, 5]
    })
    return df1, df2


@pytest.fixture
def train_data_pos():
    df1 =  pd.DataFrame({
        DEFAULT_USER_COL: [1, 2],
        DEFAULT_ITEM_COL: [3, 6]
    })
    df2 =  pd.DataFrame({
        DEFAULT_USER_COL: [1],
        DEFAULT_ITEM_COL: [3]
    })
    return  df1, df2


@pytest.fixture
def train_data_neg():
    df1 =  pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 2],
        DEFAULT_ITEM_COL: [3, 6, 7]
    })
    df2 =  pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 2, 2],
        DEFAULT_ITEM_COL: [6, 7, 3, 4]
    })
    return  df1, df2


def test_random_sample(all_data, test_data, train_data_pos, train_data_neg):
    df_test1, df_test2 = test_data
    df_train_neg_gt1, df_train_neg_gt2 = train_data_neg
    df_train1, df_train2 = train_data_pos
    df_train_neg = random_sample(df_train1, all_data, 1)
    for index, value in enumerate(df_train_neg.groupby(DEFAULT_USER_COL)[DEFAULT_ITEM_COL]):
        bool_set = set(~value[1].isin(all_data[all_data[DEFAULT_USER_COL] == value[0]][DEFAULT_ITEM_COL]))
        assert len(bool_set) == 1, 'There are items be both negative and postive'

    df_result1 = random_sample(df_test2, all_data, 3, df_train_neg_gt1, True)
    assert_series_equal(pd.Series([4, 5, 13 ]), pd.Series(df_result1[DEFAULT_ITEM_COL]), check_names=False)

    df_result2 = random_sample(df_test1, all_data, 3, df_train_neg_gt2, True)
    assert_series_equal(pd.Series([4, 5, 10,7,13, 5]), pd.Series(df_result2[DEFAULT_ITEM_COL]), check_names=False)
