from src.data_process.jobdataset import get_career
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
import os
import numpy as np
# import pandas as pd
# from cmfrec import MostPopular
import pickle


def als(model_path, user_job_app):
    user_job_app = bm25_weight(user_job_app, K1=100, B=0.8)
    # user_job_app = job_user_app.T.tocsr()
    model = AlternatingLeastSquares(factors=64, regularization=0.05)
    model.fit(2 * user_job_app)
    with open(os.path.join(model_path, 'model_als.sav'), 'wb') as pickle_out:
        pickle.dump(model, pickle_out)


def bpr(model_path, user_job_app):
    # user_job_app = job_user_app.T.tocsr()
    model = BayesianPersonalizedRanking()
    model.fit(user_job_app)
    with open(os.path.join(model_path, 'model_bpr.sav'), 'wb') as pickle_out:
        pickle.dump(model, pickle_out)


def lmf(model_path, user_job_app ):
    # user_job_app = job_user_app.T.tocsr()
    model = LogisticMatrixFactorization()
    model.fit(user_job_app)
    with open(os.path.join(model_path, 'model_lmf.sav'), 'wb') as pickle_out:
        pickle.dump(model, pickle_out)


def random_model(X_test, model_path):
    # Random recommendations (random latent factors)
    rng = np.random.default_rng(seed=1)
    UserFactors_random = rng.standard_normal(size=(X_test.shape[0], 5))
    ItemFactors_random = rng.standard_normal(size=(X_test.shape[1], 5))
    with open(os.path.join(model_path, 'model_rand.sav'), 'wb') as pickle_out:
        pickle.dump(ItemFactors_random, pickle_out)


def popular(X_train, model,math):
    # Non-personalized recommendations
    model_baseline = MostPopular(implicit=True, user_bias=False).fit(X_train.tocoo())
    item_biases = model_baseline.item_bias_
    with open(os.path.join(item_biases, 'model_pop.sav'), 'wb') as pickle_out:
        pickle.dump(model, pickle_out)


def main():
    jobs, users, job_user_app = get_career('../data/clean/apps_pos.hdf5')
    model_path = "../data/model/all_1"
    user_job_app = job_user_app.T.tocsr()
    bpr(model_path, user_job_app)
    als(model_path, user_job_app)
    lmf(model_path, user_job_app)


if __name__ == "__main__":
    main()
# userid = 12345
# ids, scores = model.recommend(userid, user_plays[userid], N=10, filter_already_liked_items=True)
# # print(ids)
# # print(scores)
#
# print(pd.DataFrame({"jobs": jobs[ids], "score": scores, "already_liked": np.in1d(ids, user_plays[userid].indices)}))













