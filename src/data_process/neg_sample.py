import os
import sys
sys.path.append('../../src')
import pandas as pd
# import modin.pandas as pd
import numpy as np
import tqdm
import random
from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL


def random_sample(data_gt, all_data, ng_num, train_neg=None, test=False):
    neg_list = []
    all_items = set(all_data[DEFAULT_ITEM_COL])
    for index, value in enumerate(tqdm.tqdm(data_gt.groupby(DEFAULT_USER_COL))):
        # for i in tqdm.tqdm(data_gt.values):
        train_user_item = set(all_data[all_data[DEFAULT_USER_COL] == value[0]][DEFAULT_ITEM_COL])
        if test:
            train_user_item_neg = set(train_neg[train_neg[DEFAULT_USER_COL] == value[0]][DEFAULT_ITEM_COL])
            item_diff = all_items - train_user_item - train_user_item_neg
        else:
            item_diff = all_items - train_user_item
        random.seed(10)
        neg = random.sample(item_diff, ng_num * len(value[1]))
        if test:
            neg_list = keep_pos_test(neg_list, value)
        for j in neg:
            neg_list.append([value[0], j, 0])
    df_neg = pd.DataFrame(neg_list, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]).reset_index(
        drop=True)
    return df_neg


def keep_pos_test(neg_list, value):
    rows = value[1].shape[0]
    ones = np.ones(rows, dtype='int8')
    re_ones = ones.reshape(rows, 1)
    nest_list = np.append(value[1].values, re_ones, axis=1).tolist()
    for i in nest_list:
        neg_list.append(i)
    return neg_list


def save_data(data, path):
    data.to_feather(path)


def read_csv(path_read):
    all_data = pd.read_csv(
        path_read,
        header=0, names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, 'Split'])
    train_data = all_data[all_data['Split'] == 'Train']
    test_gt = all_data[all_data['Split'] == 'Test'][[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
    return all_data, test_gt, train_data


def read_feather_all(path_read):
    extension = os.path.splitext(path_read)[-1]
    if extension == '.csv':
        all_data = pd.read_csv(path_read, names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, 'Split'], header=None)
    else:
        all_data = pd.read_feather(
            path_read, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, 'Split'])
    # all_data.rename(columns={DEFAULT_USER_COL: DEFAULT_USER_COL, DEFAULT_ITEM_COL: DEFAULT_ITEM_COL, 'Split': 'Split'}, inplace=True)
    all_data[DEFAULT_USER_COL].astype('int64')
    all_data[DEFAULT_ITEM_COL].astype('int64')
    train_data = all_data[all_data['Split'] == 'Train'][[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
    test_gt = all_data[all_data['Split'] == 'Test'][[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
    return all_data, test_gt, train_data


def read_feather(path_read):
    extension = os.path.splitext(path_read)[-1]
    if extension == '.csv':
        data = pd.read_csv(
            path_read, usecols=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        )
    else:
        data = pd.read_feather(
            path_read, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
    # data.rename(columns={DEFAULT_USER_COL: DEFAULT_USER_COL, DEFAULT_ITEM_COL: DEFAULT_ITEM_COL}, inplace=True)
    data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].astype('int64')
    data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].astype('int64')
    return data


def main(path_read_all, path_read_goal, path_save, ng_num, test, path_read_neg=None):
    all_data = read_feather(path_read_all)
    data = read_feather(path_read_goal)
    train_neg = None
    if test:
        train_neg = read_feather(path_read_neg)
    ng_data = random_sample(data, all_data, ng_num, train_neg, test)
    save_data(ng_data, path_save)

if __name__ == '__main__':
    # path_read_all = '../../data/jobs/merged_sub_clean.csv'
    # path_read_train = '../../data/jobs/debug/leave_one_train.csv'
    # path_save = '../../data/jobs/debug/leave_one_train_neg'
    # ng_num = 4
    # main(path_read_all=path_read_all,
    #                 path_read_goal=path_read_train,
    #                 path_save=path_save, ng_num=ng_num,
    #                 path_read_neg=None, test=False)

    path_read_test = '../../data/jobs/debug/leave_one_test.csv'
    path_read_all = '../../data/jobs/merged_sub_clean.csv'
    path_save = '../../data/jobs/debug/test_pos_neg'
    train_neg = '../../data/jobs/debug/leave_one_train_neg.csv'
    ng_num = 100
    main(path_read_all=path_read_all,
                path_read_goal=path_read_test,
                path_save=path_save,
                ng_num=ng_num,
                test=True,
                path_read_neg = train_neg)