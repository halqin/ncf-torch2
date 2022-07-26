import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
import time
from hydra import initialize, compose
import pathlib
# import config

import src.data_process.neg_sample as ng_sample
from src.data_process.utils import mix_merge
from src.data_process.data_split import data_split_user
from src.metrics.evaluate_ignite import CustomHR, CustomNDCG, CustomAuc_top, CustomAuc, CustomRecall_top, CustomPrecision_top
from src.model_entity import EntityCat
from src.data_utils import CatData
from src.utils.constants import DEFAULT_USER_COL,DEFAULT_ITEM_COL,DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL


from sklearn import metrics, preprocessing
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# import argparse
torch.manual_seed(0)

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator, RemovableEventHandle
from ignite.metrics import Accuracy, Loss, Metric
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.wandb_logger import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=[])


if device.type =='cpu':
    BATCH_SIZE = cfg.params.batch_size_cpu
    EPOCHS  = cfg.params.epochs_cpu
else:
    BATCH_SIZE = cfg.params.batch_size_gpu
    EPOCHS  = cfg.params.epochs_gpu



if device.type == 'cpu':
    use_amp=False
    df_train_pos  = ng_sample.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_pos))
    df_train_neg = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_neg))
    df_test_ori = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.test_pos_neg)).iloc[:202,]
    df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
    df_train_pos = df_train_pos.sort_values(by=[DEFAULT_USER_COL]).iloc[:100,].reset_index(drop=True)
    df_train_neg = df_train_neg.sort_values(by=[DEFAULT_USER_COL]).iloc[:100*cfg.params.neg_train,].reset_index(drop=True)
else:
    use_amp=True
    df_train_pos  = ng_sample.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_pos))
    df_train_neg = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_neg))
    df_test_ori = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.test_pos_neg))
    df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
    df_train_pos = df_train_pos.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)
    df_train_neg = df_train_neg.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)


# In[15]:


df_train_pos[DEFAULT_RATING_COL] = 1


# In[16]:


def concat_index(df1, df2):
    df2.index = df2.index//cfg.params.neg_train
    return pd.concat([df1, df2], axis=0).sort_index(kind='mregesort').reset_index(drop=True)


# In[17]:


df_train_all = concat_index(df_train_pos, df_train_neg)


# In[18]:


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

# ### Choose the features and process data for the training

# In[19]:


user_features = ['DegreeType', 'Major', 'GraduationDate']
user_features_extend = [DEFAULT_USER_COL] + user_features

item_features = []
item_features_extend =[DEFAULT_ITEM_COL] + item_features

base_features = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]


# In[20]:


df_mix_merge = mix_merge(df_all , df_all_features, user_features_extend, item_features_extend)


# In[21]:


def _cat_encode(df_data, list_f, encoder):
    for f in list_f:
        df_data[f] = encoder.fit_transform(df_data[f].astype('category').cat.codes.values)
    return df_data


# In[22]:


def _embedding_dimension(df_all_encode, features_to_train, max_dim=50):
    embedding_size = []
    features_to_em = [i for i in features_to_train if i !=DEFAULT_RATING_COL]
    for c in features_to_em:
        num_unique_values = int(df_all_encode[c].nunique())
        embed_dim = int(min(np.ceil(num_unique_values/2), max_dim))
        embedding_size.append([num_unique_values, embed_dim])  
    return embedding_size


# In[23]:


def encode_data(df_mix_merge, features_to_code, features_to_train, max_dim=50):
    encoder = preprocessing.LabelEncoder()
    df_all_encode = _cat_encode(df_mix_merge, features_to_code, encoder)
    df_train = df_all_encode[df_all.flag==1]
    df_test = df_all_encode[df_all.flag==0]
    df_train = df_train[features_to_train]
    df_test = df_test[features_to_train]
    embedding_size = _embedding_dimension(df_all_encode, features_to_train, max_dim)
    return df_train, df_test, embedding_size


# In[24]:


num_feature=[]
features_to_code = df_mix_merge.columns
features_to_train = [DEFAULT_USER_COL, DEFAULT_ITEM_COL]+ user_features + item_features +[DEFAULT_RATING_COL]


# In[25]:


df_train,  df_test, embedding_size = encode_data(df_mix_merge, features_to_code, features_to_train, max_dim=cfg.params.emb_dim)

print(f'The size of embedding layers:{embedding_size}')


# ## Run data check before training 

# Check the ratio of positive and negative samples

# In[26]:


assert len(df_train[df_train.rating==0])/len(df_train[df_train.rating==1]) == cfg.params.neg_train, 'wrong neg/pos ratio in training set'
assert len(df_test[df_test.rating==0])/len(df_test[df_test.rating==1]) == cfg.params.neg_test, 'wrong neg/pos ratio in test set '
#Check if all the users in test can be found in training set
assert sum(np.isin(df_test.userid.unique(), df_train.userid.unique(), assume_unique=True)) == len(df_test.userid.unique()), 'cold start'
#The the uniqueness of items between training and test. For a user, on common items between training and test dataset. 
assert df_all.shape[0] ==df_train.shape[0]+df_test.shape[0], 'wrong data concat'
assert sum(df_all.groupby(['userid']).apply(lambda x: len(x['itemid'].unique()))) == df_all.shape[0], 'train and test have overlap item'


# ## Creat the numpy array for training 

# In[27]:


df_train_split, df_val_split = data_split_user(df_train, val_size=0.2)

np_train = df_train_split.values
np_val = df_val_split.values
np_test = df_test.values


# In[28]:


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


# In[29]:


train_dataset = CatData(np_train)
val_dataset = CatData(np_val)
test_dataset = CatData(np_test) 
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,  worker_init_fn=seed_worker,generator=g)
val_loader = data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0,  worker_init_fn=seed_worker,generator=g)
test_loader = data.DataLoader(test_dataset, batch_size=cfg.params.neg_test+1, shuffle=False, num_workers=0,worker_init_fn=seed_worker,generator=g )


# In[30]:


model = EntityCat(embedding_size = embedding_size, num_numerical_cols = len(num_feature),
               output_size = 1)
model.to(device)


# In[31]:


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg.params.lr)


# ## Model the training 

# In[32]:


def output_trans_loss(output):
    return output['y_pred'], output['label']

val_metrics_train = {
    'auc': CustomAuc(),
    "loss": Loss(criterion, output_transform=output_trans_loss)
}

val_metrics_test = {
    'hr': CustomHR(),
    'ndcg': CustomNDCG(k=cfg.params.topk),
    'auc': CustomAuc(),
    'auc_top': CustomAuc_top(),
    'recall_top': CustomRecall_top(k=cfg.params.topk),
    'precision_top': CustomPrecision_top(k=cfg.params.topk),
    "loss": Loss(criterion, output_transform=output_trans_loss)
}


# In[40]:


scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        y_pred = model(x).reshape(1,-1).flatten()
        loss = criterion(y_pred, y.float())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, label = batch[0].to(device), batch[1].to(device)
        y_pred = model(x).reshape(1,-1).flatten()
        label=label.float()
        return {'label':label, 'y_pred':y_pred}


def test_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, label = batch[0].to(device), batch[1].to(device)
        y_pred = model(x).reshape(1,-1).flatten()
        label=label.float()
        
        y_pred_top, indices = torch.topk(y_pred, engine.state.topk)
        
        y_pred_top = y_pred_top.detach().cpu().numpy()
        reco_item = torch.take(x[:,1], indices).cpu().numpy().tolist()
        pos_item = x[0,1].cpu().numpy().tolist()  # ground truth, item id
        label_top = label[indices].cpu().numpy()
        indices = indices.cpu().numpy()
        return {'pos_item':pos_item, 'reco_item':reco_item, 'y_pred_top':y_pred_top, 
                'label_top':label_top, 'label':label, 'y_pred':y_pred, 'y_indices':indices}
    
trainer = Engine(train_step)

train_evaluator = Engine(validation_step)
# train_evaluator.state_dict_user_keys.append('topk')

val_evaluator = Engine(validation_step)
# val_evaluator.state_dict_user_keys.append('topk')

test_evaluator = Engine(test_step)
test_evaluator.state_dict_user_keys.append('topk')

# @val_evaluator.on(Events.STARTED)
# def init_user_value():
#     val_evaluator.state.topk=3
    
# @train_evaluator.on(Events.STARTED)
# def init_user_value():
#     train_evaluator.state.topk=3

@train_evaluator.on(Events.STARTED)
def init_user_value():
    test_evaluator.state.topk=cfg.params.topk
    
    
# Attach metrics to the evaluators
for name, metric in val_metrics_train.items():
    metric.attach(train_evaluator, name)

for name, metric in val_metrics_train.items():
    metric.attach(val_evaluator, name)

    
for name, metric in val_metrics_test.items():
    metric.attach(test_evaluator, name)
    
# Eearly_stop 
def score_function(engine):
    val_loss = engine.state.metrics['auc']
    return val_loss

Eearly_stop_handler = EarlyStopping(patience=cfg.params.patience, score_function=score_function, trainer=trainer)
# val_evaluator.add_event_handler(Events.COMPLETED, Eearly_stop_handler)


# In[41]:


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    auc = metrics['auc']
    loss = metrics['loss']
    print(f'Training Results- Epoch[{trainer.state.epoch}]  Avg loss: {loss:.2f}           Avg auc:{auc:.2f}')


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(val_loader)
    metrics = val_evaluator.state.metrics
    auc = metrics['auc']
    loss = metrics['loss']
    print(f'Validation Results- Epoch[{trainer.state.epoch}]  Avg loss: {loss:.2f}           Avg auc:{auc:.2f}')

    
@trainer.on(Events.COMPLETED)
def log_test_results(trainer):
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    hr = metrics['hr']
    ndcg = metrics['ndcg']
    auc = metrics['auc']
    auc_top = metrics['auc_top']
    recall = metrics['recall_top']
    precision = metrics['precision_top']
    loss = metrics['loss']
    print(f"Test Results - Epoch[{trainer.state.epoch}]  Avg loss: {loss:.2f}      Avg ndcg: {ndcg:.2f}  Avg auc: {auc:.2f}  Avg auc_top: {auc_top:.2f}       Avg recall: {recall:.2f}  Avg precision: {precision:.2f}")

pbar = ProgressBar(persist=False)
pbar.attach(trainer)


# In[42]:


trainer.run(train_loader, max_epochs=2)


# In[36]:


config_dict = dict(cfg.params)
config_dict['Features']='-'.join(user_features+item_features)
wandb_logger = WandBLogger(
    project="pytorch-jrs",
    name="-".join(user_features)+'-'+'-'.join(item_features),
    config=config_dict,
    tags=["entity", "jrs"]
)

to_save = {'model': model}
checkpoint_handler = ModelCheckpoint(
    wandb_logger.run.dir,
    n_saved=1, filename_prefix='best',
    score_name="auc",
    global_step_transform=global_step_from_engine(trainer)
)

val_evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, to_save)

    
wandb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    output_transform=lambda loss: {"loss": loss}
)

wandb_logger.attach_output_handler(
    train_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names=['loss','auc'],
    global_step_transform=lambda *_: trainer.state.iteration,
)

wandb_logger.attach_output_handler(
    val_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names=['loss',"auc"],
    global_step_transform=lambda *_: trainer.state.iteration,
)


wandb_logger.attach_output_handler(
    test_evaluator,
    event_name=Events.COMPLETED,
    tag="test",
    metric_names=['loss',"auc", 'hr', 'ndcg', 'auc_top', 'recall_top', 'precision_top'],
    global_step_transform=lambda *_: trainer.state.iteration,
)


wandb_logger.attach_opt_params_handler(
    trainer,
    event_name=Events.ITERATION_STARTED,
    optimizer=optimizer,
    param_name='lr'  # optional
)

# wandb_logger.watch(model) 

# trainer.run(train_loader, max_epochs=EPOCHS)
wandb_logger.close()


