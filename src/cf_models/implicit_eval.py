import os.path
import pickle
import pandas as pd
import logging
import numpy as np
from datasets.jobdataset import get_career
from implicit.evaluation import AUC_at_k, ndcg_at_k, train_test_split, precision_at_k, mean_average_precision_at_k
from implicit_build import als, bpr, lmf
import recometrics
from scipy.sparse import coo_matrix, csr_matrix
from recommenders.evaluation.python_evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
)
from utils.constants import DEFAULT_USER_COL,DEFAULT_ITEM_COL,DEFAULT_RATING_COL, DEFAULT_PREDICTION_COL



def quantity_eval(userid, logger, user_info, job_info, user_app, model, user_plays, jobs):
    user_info_single = user_info.Major[user_info[DEFAULT_USER_COL] == userid].to_numpy()
    user_job_hist = user_app.JobID[user_app[DEFAULT_USER_COL] == userid]
    job_title_hist = job_info.Title[job_info.JobID.isin(user_job_hist)].to_numpy()
    ids, scores = model.recommend(userid, user_plays[userid], N=10, filter_already_liked_items=True)
    job_title_rec = job_info.Title[job_info.JobID.isin(jobs[ids])].to_numpy()
    logger.info(f'The user info is: {user_info_single}')
    logger.info(f'The user\'s history applications are: {job_title_hist}')
    logger.info(f'The recommendation list is: {job_title_rec}')
    return None


def quality_eval(model, train, test, int_k, logger):
    p_at_k = precision_at_k(model, train, test, K=int_k)
    m_at_k = mean_average_precision_at_k(model, train, test, K=int_k)
    ndcg = ndcg_at_k(model, train, test, K=int_k)
    AUC = AUC_at_k(model, train, test, K=int_k)
    logger.info(f'Precision: {p_at_k}, Mean_average_precision: {m_at_k}, NDCG: {ndcg}, AUC:{AUC}')
    return p_at_k, m_at_k, ndcg, AUC


def create_logger(log_path, log_name):
    # create logger
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    # create console handler and set level to debug
    fh = logging.FileHandler(os.path.join(log_path, log_name + '.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s --- \n %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def model_build(model_path, train=None, test=None):
    bpr(model_path, train)
    # als(model_path, train)
    lmf(model_path, train)
    # popular(model_path, train)
    # random_model(model_path, test)
    return None


def recometrics_eval(model, train, test, k, logger):
    metrics = recometrics.calc_reco_metrics(
        train[:test.shape[0]], test,
        model.user_factors[:test.shape[0]], model.item_factors,
        k=k,
        roc_auc=True,
        ndcg=True,
        average_precision=True,
        precision=True,
        recall=True,
        hit=True
    )

    logger.info(metrics.mean(axis=0).to_frame().T)
    return None


def microsoft_eval(model, train, test_gddf, usersid, jobsid, k, logger):
    # test_df = sparse_todf(test_sparse, usersid, jobsid, p)
    predict_df = model_infer_df(test_gddf, usersid, jobsid, model, train, k)

    pk = precision_at_k(
        rating_true=test_gddf,
        rating_pred=predict_df,
        col_prediction=DEFAULT_RATING_COL,
        k=k,
    )
    rk = recall_at_k(
        rating_true=test_gddf,
        rating_pred=predict_df,
        col_prediction=DEFAULT_RATING_COL,
        k=k,
    )
    ndcgk = ndcg_at_k(
        rating_true=test_gddf,
        rating_pred=predict_df,
        col_prediction=DEFAULT_RATING_COL,
        k=k,
    )
    mapk = map_at_k(
        rating_true=test_gddf,
        rating_pred=predict_df,
        col_prediction=DEFAULT_RATING_COL,
        k=k,
    )
    # auck = auc(
    #     rating_true=test_gddf,
    #     rating_pred=test_df,
    #     col_prediction=DEFAULT_RATING_COL,
    # )
    auck = 1
    logger.info(f'The precision@k is: {pk}; recall@k is: {rk};The NDCG@k is: {ndcgk};MAP@k is: {mapk}; AuC@k is: {auck}')
    return None


def sparse_todf(sparse, usersid, jobsid, path):
    ''' Convert the sparse matrix to dataframe
    :param sparse: The sparse matrix of train/test set
    :param usersid:
    :param jobsid:
    :return:
    '''
    nonzerolen = sparse.count_nonzero()
    out_matrix = np.empty((nonzerolen, 2), dtype=int)
    for i in range(nonzerolen):
        user_ = sparse.nonzero()[0][i]
        job_ = sparse.nonzero()[1][i]
        out_matrix[i][0] = usersid[user_]
        out_matrix[i][1] = jobsid[job_]
    out_df = pd.DataFrame(out_matrix, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
    out_df[DEFAULT_USER_COL] = out_df[DEFAULT_USER_COL].astype('str')
    out_df[DEFAULT_ITEM_COL] = out_df[DEFAULT_ITEM_COL].astype('str')
    out_df[DEFAULT_RATING_COL] = 1
    out_df = out_df.rename(columns={DEFAULT_USER_COL: DEFAULT_USER_COL, DEFAULT_ITEM_COL: DEFAULT_ITEM_COL})
    out_df.to_csv(path, index=False)
    return out_df


def model_infer_df(apps_true, usersid, jobsid, model, train, k):
    '''Convert the model prediction to dataframe
    :param apps_true:
    :param usersid:
    :param model:
    :param train:
    :param k:
    :return:
    '''
    reco_jobsid = []
    test_users = list(set(apps_true[DEFAULT_USER_COL]))
    test_user_indexes = list(np.searchsorted(usersid, test_users))
    ids, scores = model.recommend(test_user_indexes, train[test_user_indexes],
                                  N=k, filter_already_liked_items=True)


    reco_jobs_index = np.reshape(ids, (len(test_user_indexes) * k, 1))
    reco_prob = np.reshape(scores, (len(test_user_indexes) * k, 1))
    for i in reco_jobs_index.flatten():
        reco_jobsid.append(jobsid[i])
    reco_users = np.repeat(test_users, k)
    results = pd.DataFrame()
    results[DEFAULT_USER_COL] = reco_users
    results[DEFAULT_ITEM_COL] = reco_jobsid
    results[DEFAULT_RATING_COL] = 1
    results[DEFAULT_PREDICTION_COL] = reco_prob

    results[DEFAULT_USER_COL] = results[DEFAULT_USER_COL].astype('str')
    results[DEFAULT_ITEM_COL] = results[DEFAULT_ITEM_COL].astype('str')
    # results.to_csv('../data/clean/sub/predict_result.csv')
    return results


def toy():
    toy = pd.read_csv('toy.csv')
    toy[DEFAULT_USER_COL] = pd.Categorical(toy.UserId).codes
    toy[DEFAULT_ITEM_COL] = pd.Categorical(toy.ItemId).codes
    X = coo_matrix((toy.Count, (toy.UserId, toy.ItemId)))
    return X


def customer_split():
    filename = "../data/clean/apps_pos_neg_newsplit"
    data = pd.read_feather(
        filename, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, "Split"]
    )
    df_pos = data[data[DEFAULT_RATING_COL] == 1].copy()
    df_pos[DEFAULT_USER_COL] = pd.Categorical(df_pos.UserID).codes
    df_pos[DEFAULT_ITEM_COL] = pd.Categorical(df_pos.JobID).codes
    x = coo_matrix((df_pos[DEFAULT_RATING_COL], (df_pos.UserID, df_pos.JobID)))
    test_index = df_pos['Split'] == 'Test'
    train_index = df_pos['Split'] == 'Train'
    train_csr = csr_matrix((x.data[train_index],
                            (x.row[train_index], x.col[train_index])),
                           shape=x.shape, dtype=x.dtype)
    test_csr = csr_matrix((x.data[test_index],
                           (x.row[test_index], x.col[test_index])),
                          shape=x.shape, dtype=x.dtype)
    return train_csr, test_csr


def read_train_gd_csv(data_testgd_path):
    test_gddf = pd.read_csv(data_testgd_path)
    test_gddf[DEFAULT_USER_COL] = test_gddf[DEFAULT_USER_COL].astype('str')
    test_gddf[DEFAULT_ITEM_COL] = test_gddf[DEFAULT_ITEM_COL].astype('str')
    return test_gddf

def main():
    k = 10
    model_path = "../data/model/sub"
    log_path = '../data/log'
    log_name = 'toy'
    data_raw_name = 'apps_pos_sub.hdf5'
    test_gd_name = "test_gd_df.csv"
    data_path = "../data/clean/sub"
    logger = create_logger(log_path, log_name)
    data_raw_path = os.path.join(data_path, data_raw_name)
    data_testgd_path = os.path.join(data_path, test_gd_name)
    # job_info = pd.read_feather('../data/clean/jobs')
    # user_info = pd.read_feather('../data/clean/users')
    # user_app = pd.read_feather('../data/clean/apps')
    # jobs, users, job_user_train = get_career('../data/clean/apps_pos_newsplit_train.hdf5')
    # _, _, job_user_apply_test = get_career('../data/clean/apps_pos_newsplit_test.hdf5')
    jobsid, usersid, job_user_apply = get_career(data_raw_path)
    train, test = train_test_split(job_user_apply, train_percentage=0.6, random_state=123)
    # train, test = job_user_train, job_user_apply_test
    # train, test = customer_split()

    model_build(model_path, train=train, test=test)
    # sparse_todf(test, usersid, jobsid, data_testgd_path)
    test_gddf = read_train_gd_csv(data_testgd_path)

    algorithm_dic = {
        # 'als': 'model_als.sav',
        # 'bpr': 'model_bpr.sav',
        # 'lmf': 'model_lmf.sav',
        # 'pop': 'model_pop.sav',
        # 'rand': 'model_rand.sav'
    }
    algorithm_dic = {'regression': 'model_regression.sav'}
    for item in algorithm_dic.items():
        logger.info(f'========== The algorithm is: {item[0]} ==========')
        with open(os.path.join(model_path, item[1]), 'rb') as pickle_in:
            model = pickle.load(pickle_in)
        # pak = precision_at_k(model, train, test, K=10)
        # quantity_eval()
        # quality_eval(model, train, test, int_k=k, logger=logger)
        # recometrics_eval(model, train, test, k, logger)
        microsoft_eval(model, train, test_gddf=test_gddf, usersid=usersid, jobsid=jobsid, k=10, logger=logger)


if __name__ == "__main__":
    main()
