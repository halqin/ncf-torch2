import pandas as pd
# import sys
import os
import numpy as np
# sys.path.append('../../src')
from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL


def _load_all(path_read):
    extension = os.path.splitext(path_read)[-1]
    if extension == '.csv':
        all_data = pd.read_csv(path_read, low_memory=False)
    else:
        all_data = pd.read_feather(path_read)
    return all_data


def _leave_one_out(data, path1, path2):
    data_sorted = data.sort_values([DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL])
    test = data_sorted.groupby(DEFAULT_USER_COL).last().reset_index()
    train = data_sorted[~data_sorted[DEFAULT_TIMESTAMP_COL].isin(test[DEFAULT_TIMESTAMP_COL])].reset_index()
    test = test[test[DEFAULT_USER_COL].isin(train[DEFAULT_USER_COL].unique())]
    test = test[test[DEFAULT_ITEM_COL].isin(train[DEFAULT_ITEM_COL].unique())].reset_index()
    train.to_csv(path1, index=False)
    test.to_csv(path2, index=False)
    # return  train, test


def main(path_read, path1, path2):
    all_data = _load_all(path_read)
    _leave_one_out(all_data, path1, path2)


def data_split_user(df_train, val_size=0.2):
    unique_user = df_train[DEFAULT_USER_COL].unique()
    val_user = np.random.choice(unique_user, int(val_size*len(unique_user)), replace=False)
    df_train_split = df_train[~(df_train[DEFAULT_USER_COL].isin(val_user))]
    df_val_split = df_train[df_train[DEFAULT_USER_COL].isin(val_user)]
    return df_train_split, df_val_split

if __name__ == "__main__":
    path_read = "../../data/jobs/merged_sub.csv"
    path_save1 = "../../data/jobs/leave_one_train.csv"
    path_save2 = "../../data/jobs/leave_one_test.csv"
    main(path_read, path_save1, path_save2)
