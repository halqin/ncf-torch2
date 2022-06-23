import os

from src.data_utils import load_all
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

@pytest.fixture
def train_data_gt():
    return ([[0, 0], [0, 1], [1, 0], [2, 2],
             [2, 3], [3, 0], [3, 4], [3, 5]])


@pytest.fixture
def test_data_gt():
    return ([[0, 2], [0, 3], [0, 5], [0, 2],
             [1, 2], [1, 4], [1, 1], [1, 3],
             [2, 5], [2, 4], [2, 3], [2, 5],
             [3, 1], [3, 2], [3, 5], [3, 4]])


def test_load_all(train_data_gt, test_data_gt):
    user_num_gt = 4
    item_num_gt = 7
    train_data, test_data, user_num, item_num, train_mat = load_all()
    np.testing.assert_array_equal(train_data, train_data_gt)
    # print(test_data)
    np.testing.assert_array_equal(test_data, test_data_gt)
    assert user_num_gt == user_num
    assert item_num_gt == item_num

