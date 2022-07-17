#!/usr/bin/env python
# coding: utf-8

# In[41]:


import sys
sys.path.append('../src')
import pandas as pd
import numpy as np
# import config
import data_process.neg_sample as ng_sample
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import random
from metrics import evaluate_ignite
from model_entity import EntityCat
from data_utils import CatData
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
from utils.constants import DEFAULT_USER_COL,DEFAULT_ITEM_COL,DEFAULT_RATING_COL

# import argparse
torch.manual_seed(0)


# In[2]:


BATCH_SIZE = 10
EPOCHS  = 10
TOP_K = 10


# In[3]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# In[4]:


device


# In[5]:


df_train1  = ng_sample.read_feather("../../data/jobs/leave_one_train.csv").iloc[:100,]
df_train2 = pd.read_feather("../data/jobs/leave_one_train_neg").iloc[:100, ]
df_test_ori = pd.read_feather("../data/jobs/test_pos_neg").iloc[:101, ]
df_all_features = pd.read_csv('../data/jobs/merged_sub_clean.csv')


# In[6]:


df_train1['rating'] = 1
df_train_all = pd.concat([df_train1, df_train2], axis=0)
df_train_all['flag'] = 1
df_test_ori['flag'] = 0
df_all = pd.concat([df_train_all, df_test_ori], axis=0).reset_index(drop=True)


# user features: 
#        'WindowID_user', 'Split', 'City',
#        'State', 'Country', 'Zip_user', 'DegreeType', 'Major', 'GraduationDate',
#        'WorkHistoryCount', 'TotalYearsExperience', 'CurrentlyEmployed',
#        'ManagedOthers', 'ManagedHowMany',
#        
# job features: 
#        'WindowID_job', 'City_job',
#        'State_job', 'Country_job', 'Zip_job', 'StartDate', 'EndDate',

# ### Choose the features for the model

# In[7]:


user_features = ['City']
user_features_extend = [DEFAULT_USER_COL] + user_features

item_features = ['City_job']
item_features_extend =[DEFAULT_ITEM_COL] + item_features

base_features = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, ]


# In[8]:


def unique_filter(df_data, name_col):
    df_uni=df_data[~(df_data[name_col].duplicated())].reset_index(drop=True)
    return df_uni


# In[9]:


def feature_merge(list_f, df_all, df_all_f, default_x_col):
    df_x_features = df_all_f[list_f]
    df_x_unique = unique_filter(df_x_features, default_x_col)
    df_merge_x = df_all.merge(df_x_unique, how='left', on=[default_x_col])
    return df_merge_x


# In[10]:


def user_item_merge(mode, df_all, df_all_features, feature_list):
    if mode =='user':
        df_merge_x = feature_merge(feature_list, df_all, df_all_features,                                      DEFAULT_USER_COL)
    if mode == 'item':
        df_merge_x = feature_merge(feature_list, df_all, df_all_features,                                    DEFAULT_ITEM_COL)

    return df_merge_x


# In[11]:


def mix_merge(df_all, df_all_features, f_list_user, f_list_item):
    df_merge_user = feature_merge(f_list_user, df_all, df_all_features,                                   DEFAULT_USER_COL)
    df_merge_item = feature_merge(f_list_item, df_all, df_all_features,                                   DEFAULT_ITEM_COL)
    df_merge_x = pd.concat([df_merge_user[f_list_user],                              df_merge_item[f_list_item]], axis=1)
    df_merge_x[DEFAULT_RATING_COL] = df_all[DEFAULT_RATING_COL]
    assert df_all.shape[0] == df_merge_x.shape[0], "wrong merge"
    return df_merge_x


# In[12]:


df_mix_merge = mix_merge(df_all, df_all_features, user_features_extend, item_features_extend)


# In[13]:


le = preprocessing.LabelEncoder()


# In[14]:


def cat_encode(df_data, list_f):
    for f in list_f:
        df_data[f] = le.fit_transform(df_data[f].astype('category').cat.codes.values)
    return df_data


# In[15]:


features_to_code = df_mix_merge.columns


# In[16]:


df_all_encode = cat_encode(df_mix_merge, features_to_code)


# In[17]:


# df_all_encode[DEFAULT_RATING_COL] = df_all[DEFAULT_RATING_COL]


# In[18]:


df_train = df_all_encode[df_all.flag==1]
df_test = df_all_encode[df_all.flag==0]

# df_train=df_train.drop(['flag'], axis=1)
# df_test=df_test.drop(['flag'], axis=1)


# In[19]:


features_to_train = [DEFAULT_USER_COL, DEFAULT_ITEM_COL]+ user_features + item_features +[DEFAULT_RATING_COL]
df_train = df_train[features_to_train]
df_test = df_test[features_to_train]


# In[20]:


tb_cf = "-".join(user_features)+'-'.join(item_features)
tb_cf


# In[21]:


df_train=df_train[features_to_train]
df_test=df_test[features_to_train]

num_feature=[]
features_to_train.remove(DEFAULT_RATING_COL)
# label_name = DEFAULT_RATING_COL


# In[22]:


np_train = df_train.values
np_test = df_test.values


# In[43]:


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


# In[44]:


train_dataset = CatData(np_train)
test_dataset = CatData(np_test) 
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,  worker_init_fn=seed_worker,generator=g)
test_loader = data.DataLoader(test_dataset, batch_size=100+1, shuffle=False, num_workers=0,worker_init_fn=seed_worker,generator=g )


# In[45]:


embedding_size = []
for c in features_to_train:
    num_unique_values = int(df_all_encode[c].nunique())
    embed_dim = int(min(np.ceil(num_unique_values/2), 50))
    embedding_size.append([num_unique_values, embed_dim])  


# In[46]:


model = EntityCat(embedding_size = embedding_size, num_numerical_cols = len(num_feature),
               output_size = 2)
model.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[47]:


print(model)


# In[48]:


timestamp = datetime.now().strftime('%m-%d_%H-%M-%S')
writer = SummaryWriter('runs/trainer_{}_{}'.format(tb_cf, timestamp))
plot_n_batch = 10



def run_one_epoch(model, epoch_index, writer, data_loader=train_loader, is_train=True):
    running_loss = 0.
    avg_loss = 0.
    HR, NDCG, ROC, ROC_top = [], [], [], []
    
    for batch, (cat_data, label) in enumerate(data_loader):
        cat_data = cat_data.to(device)
        label = label.to(device).float()
        prediction = model(cat_data)[:,1]
        loss = loss_function(prediction, label)
        running_loss += loss.item()
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        HR_1batch, NDCG_1batch, ROC_1batch, ROC_top1batch =         evaluate_entity.metrics(cat_data, prediction, label, TOP_K, is_train)
        HR.append(HR_1batch)
        NDCG.append( NDCG_1batch)
        ROC.append( ROC_1batch)
        ROC_top.append(ROC_top1batch)
            
    avg_loss = running_loss / (batch +1)  
 
    avg_HR = np.mean(HR)
    avg_NDCG = np.mean(NDCG)
    avg_ROC = np.mean(ROC)
    avg_ROC_top = np.mean(ROC_top)

    return avg_loss, avg_HR, avg_NDCG , avg_ROC, avg_ROC_top

 
for epoch in range(EPOCHS):
    print('EPOCH {}/{}: ---------'.format(epoch, EPOCHS))
    start_time = time.time()
    # Make sure gradient tracking is on, and do a pass over the data
 
    model.train(True)
    avg_loss_train, avg_HR_train, avg_NDCG_train , avg_ROC_train, avg_ROC_top_train = run_one_epoch(model,epoch,                                                         writer, data_loader=train_loader, is_train=True)
    model.train(False)
    avg_loss_test, avg_HR_test, avg_NDCG_test, avg_ROC_test, avg_ROC_top_test = run_one_epoch(model,epoch,                                                         writer, data_loader=test_loader, is_train=False)
    
    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    
    print(f"train_loss:{avg_loss_train}\ntrain_HR:{avg_HR_train}            \ntrain_NDCG:{avg_NDCG_train}\ntrain_ROC:{avg_ROC_train}\ntrain_ROC_top:{avg_ROC_top_train}            \ntest_loss:{avg_loss_test}\ntest_HR:{avg_HR_test}            \ntest_NDCG:{avg_NDCG_test}\ntest_ROC:{avg_ROC_test}" )
 
    writer.add_scalars('Loss',
                    { 'Train' : avg_loss_train, 'Test' : avg_loss_test },
                    epoch )
 
    writer.add_scalars('HitRate',
                { 'Train' : avg_HR_train, 'Test' : avg_HR_test },
                epoch)
    
    writer.add_scalars('NDCG',
            { 'Train' : avg_NDCG_train, 'Test' : avg_NDCG_test },
            epoch)
    
    writer.add_scalars('ROC',
            { 'Train' : avg_ROC_train, 'Test' : avg_ROC_test },
            epoch)

# # model_path = 'runs/model_{}'.format(timestamp)
# # torch.save(model.state_dict(), model_path)

writer.flush()
writer.close()


# In[ ]:









