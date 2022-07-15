import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# from tensorboardX import SummaryWriter
import tqdm
import data_utils
import model_entity
from metrics import evaluate_ignite
from sklearn import preprocessing
import config
import data_process.neg_sample as ng_sample

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="learning rate")
parser.add_argument("--dropout",
                    type=float,
                    default=0.0,
                    help="dropout rate")
parser.add_argument("--batch_size",
                    type=int,
                    default=2 ** 25,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=20,
                    help="training epoches")
parser.add_argument("--top_k",
                    type=int,
                    default=10,
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=32,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
                    type=int,
                    default=2,
                    help="number of neurons_in_layers in MLP model")
parser.add_argument("--num_ng",
                    type=int,
                    default=4,
                    help="sample negative items for training")
parser.add_argument("--test_num_ng",
                    type=int,
                    default=100,
                    help="sample part of negative items for testing")
parser.add_argument("--out",
                    default=True,
                    help="save model or not")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
# device = 'cpu'
print(device)

df_train1 = ng_sample.read_feather("../Data/jobs/leave_one_train")
df_train2 = pd.read_feather("../Data/jobs/pos_neg_train_uncode0")
df_test = pd.read_csv("../Data/jobs/apps_neg.csv")

df_train1=df_train1.iloc[:100, :].copy()
df_train2=df_train2.iloc[:100, :].copy()
df_test=df_test.iloc[:101, :].copy()


df_train1['rating'] = 1
df_train_all = pd.concat([df_train1, df_train2], axis=0)
df_train_all['flag'] = 1
df_test['flag'] = -1
df_all = pd.concat([df_train_all, df_test], axis=0)

le = preprocessing.LabelEncoder()
features = df_all.columns
features = [i for i in features if i not in ['flag', 'rating']]
for f in features:
    df_all[f] = le.fit_transform(df_all[f])

df_train = df_all[df_all.flag == 1]
df_test = df_all[df_all.flag == -1]

df_train = df_train.drop(['flag'], axis=1)
df_test = df_test.drop(['flag'], axis=1)

# df_train_all

num_feature = []
cat_feature = features
label_name = 'rating'

np_train = df_train.values
np_test = df_test.values

train_dataset = data_utils.CatData(np_train)
test_dataset = data_utils.CatData(np_test)
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset, batch_size=args.test_num_ng + 1, shuffle=False, num_workers=0)

embedding_size = []
for c in cat_feature:
    num_unique_values = int(df_all[c].nunique())
    embed_dim = int(min(np.ceil(num_unique_values / 2), 50))
    embedding_size.append([num_unique_values, embed_dim])

model = model_entity.EntityCat(embedding_size=embedding_size, num_numerical_cols=len(num_feature),
                               output_size=2)
model.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_hr = 0
for epoch in tqdm.tqdm(range(args.epochs)):
    model.train()
    start_time = time.time()
    for batch, (cat_data, label) in enumerate(train_loader):
        #         print('hao----', cat_data)
        cat_data = cat_data.to(device)
        label = label.to(device)
        model.zero_grad()
        prediction = model(cat_data)[:, 1]
        label = label.float()
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
    model.eval()
    HR, NDCG, ROC = evaluate_entity.metrics(model, test_loader, args.top_k)

    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
          time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

    if HR > best_hr:
        best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
        if args.out:
            if not os.path.exists(config.model_path):
                os.mkdir(config.model_path)
            torch.save(model,
                       '{}{}.pth'.format(config.model_path, config.model))

# print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
#     best_epoch, best_hr, best_ndcg))
