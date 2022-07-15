import pandas as pd
from utils.constants import DEFAULT_USER_COL,DEFAULT_ITEM_COL,DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL


def _unique_filter(df_data, name_col):
    df_uni=df_data[~(df_data[name_col].duplicated())].reset_index(drop=True)
    return df_uni



def _feature_merge(list_f, df_all, df_all_f, default_x_col):
    df_x_features = df_all_f[list_f]
    df_x_unique = _unique_filter(df_x_features, default_x_col)
    df_merge_x = df_all.merge(df_x_unique, how='left', on=[default_x_col])
    return df_merge_x


def mix_merge(df_all, df_all_features, f_list_user, f_list_item):
    df_merge_user = _feature_merge(f_list_user, df_all, df_all_features, \
                                  DEFAULT_USER_COL)
    df_merge_item = _feature_merge(f_list_item, df_all, df_all_features, \
                                  DEFAULT_ITEM_COL)
    df_merge_x = pd.concat([df_merge_user[f_list_user], \
                             df_merge_item[f_list_item]], axis=1)
    df_merge_x[DEFAULT_RATING_COL] = df_all[DEFAULT_RATING_COL]
    assert df_all.shape[0] == df_merge_x.shape[0], "wrong merge"
    return df_merge_x