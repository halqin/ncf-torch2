import pandas as pd
# import modin.pandas as pd
# import ray
# ray.init()
# from sklearn import metrics, preprocessing
import neg_sample

data_ori = neg_sample.read_feather("../../Data/jobs/leave_one_train")
test_ori = neg_sample.read_feather("../../Data/jobs/leave_one_test")
test_neg = pd.read_csv("../../Data/jobs/apps_neg.csv")

data_ori['rating'] = 1
# data_ori['flag'] = 1
# test_neg['flag'] = -1
#
df_all = pd.concat([data_ori, test_neg], axis=0)
#
# le = preprocessing.LabelEncoder()
#
# features = data_ori.columns
# features = [i for i in features if i not in ['flag', 'rating']]
#
# for f in features:
#     df_all[f] = le.fit_transform(df_all[f])
#
# df_train = df_all[df_all.flag==1]
# df_test = df_all[df_all.flag==-1]
#
# df_train=df_train.drop(['flag'], axis=1)
# df_test=df_test.drop(['flag'], axis=1)
#
# df_test.loc[df_test.groupby('user')['rating'].head(1).index, 'rating'] = 1

df_train = data_ori

neg_sample.random_sample(test_gt=df_train, all_data=df_all, ng_num=4,
                         path_save='../../Data/jobs/pos_neg_train_uncode', train=True)
