import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.utils.data as data
from collections import OrderedDict
# from src import config
from data_process.neg_sample import read_feather
import torch
torch.manual_seed(0)
np.random.seed(0)

# def load_all_csv():
# 	'''
# 	read the csv and split train, test doc
# 	:return:
# 	'''
# 	df_data = pd.read_csv()
#

def _x2id(data):
    x2id = OrderedDict()
    for i in data:
        items = x2id.keys()
        if i not in items:
            x2id[i] = len(x2id)
    return x2id


# def _item2id(data_train, data_test):
#     x2id = OrderedDict()
#     for item in data_train:
#         items = x2id.keys()
#         if item[1] not in items:
#             x2id[item[1]] = len(x2id)
#
#     for item in data_test:
#         items = x2id.keys()
#         if item[1] not in items:
#             x2id[item[1]] = len(x2id)
#     return x2id


# def _user2id(data_train, data_test):
#     x2id = OrderedDict()
#     for item in data_train:
#         items = x2id.keys()
#         if item[0] not in items:
#             x2id[item[0]] = len(x2id)
#
#     for item in data_test:
#         items = x2id.keys()
#         if item[0] not in items:
#             x2id[item[0]] = len(x2id)
#     return x2id


def _cold_start(train, test):
    ''''
    Check the cold start in the dataset.
    '''
    unique_train = np.unique([i[1] for i in train])
    unique_test = np.unique([i[1] for i in test])
    # assert np.testing.assert_array_equal(unique_test, unique_train)


def _max_check(train, test):
    max_train = max([i[1] for i in train])
    max_test = max([i[1] for i in test])
    assert max_train >= max_test, 'The training set need more element than the test set'
    return max_train, max_test


def _read_txt(config):
    '''
    Read the special format of test negative
    :return:
    '''
    test_data = []
    with open(config.test_negative, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return test_data


def _combine(data1, data2):
    return set(data1).union(set(data2))


def _read_original(config):
    train_data = pd.read_csv(
        config.train_rating,
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    test_data = _read_txt(config)
    return train_data, test_data


def _read_neg(path):
    test_data = pd.read_csv(path)
    return test_data


def load_all(test_num=100):
    """ We load all the three file here to save time in each epoch. """
    # train_data = pd.read_csv(
    #     config.train_rating,
    #     sep='\t', header=None, names=['user', 'item'],
    #     usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    #
    # # test_data_pos = pd.read_csv(
    # #     config.rating,
    # #     sep='\t', header=None, names=['user', 'item'],
    # #     usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
    #
    # test_data = _read_txt(config)

    # train_data, test_data = _read_original(config)
    # _, _, train_data = read_feather("../Data/jobs/apps_sub")
    train_data = read_feather("../Data/jobs/leave_one_train")
    # train_data['UserID'].astype('int64')
    # train_data['JobID'].astype('int64')
    test_data = _read_neg("../Data/jobs/apps_neg.csv")
    test_data = test_data.values.tolist()

    test_item = [i[1] for i in test_data]
    train_item = set(train_data['item'])
    item_all = _combine(test_item, train_item)

    test_user = [i[0] for i in test_data]
    train_user = set(train_data['user'])
    user_all = _combine(test_user, train_user)

    item2id = _x2id(item_all)
    user2id = _x2id(user_all)

    # user_num = train_data['user'].max() + 1
    user_num = len(user_all)
    item_num = len(item_all)
    train_data = train_data.values.tolist()
    # test_data_pos = test_data_pos.values.tolist()
    # item2id = _item2id(train_data, test_data_pos)
    # user2id = _user2id(train_data, test_data_pos)

    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    train_data_seq = []
    for item in train_data:
        x = user2id.get(item[0])
        y = item2id.get(item[1])
        train_data_seq.append([x, y])
        train_mat[x, y] = 1.0

    test_data_seq = []
    for item in test_data:
        x = user2id.get(item[0])
        y = item2id.get(item[1])
        test_data_seq.append([x, y])
        # train_mat[x, y] = 1.0

    # _cold_start(train_data_seq, test_data)
    _max_check(train_data_seq, test_data_seq)
    return train_data_seq, test_data_seq, user_num, item_num, train_mat


class NCFData(data.Dataset):
    def __init__(self, features,
                 num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng  # "sample negative items for training"
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:  # avoid take existing positive sample as negative sample
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
            else self.features_ps
        labels = self.labels_fill if self.is_training \
            else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item, label


class CatData(data.Dataset):
    '''
    num_f: the numerical features
    cat_f: the categorical features
    '''

    def __init__(self, inp_data):
        self.inp_data = inp_data

    def __len__(self):
        return self.inp_data.shape[0]

    def __getitem__(self, idx):
        cat_data = self.inp_data[idx, 0:-1]
        #         print(cat_data)
        label = self.inp_data[idx, -1]
        return cat_data, label
