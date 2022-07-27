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
from src.metrics.evaluate_ignite import CustomHR, CustomNDCG, CustomAuc_top, CustomAuc, CustomRecall_top, \
    CustomPrecision_top, CustomAuc_new, CustomAuc_top_new
from src.model_entity import EntityCat, EntityCat_sbert
from src.data_utils import CatData
from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL

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
from abc import ABC, abstractmethod

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=[])

if device.type == 'cpu':
    BATCH_SIZE = cfg.params.batch_size_cpu
    EPOCHS = cfg.params.epochs_cpu
else:
    BATCH_SIZE = cfg.params.batch_size_gpu
    EPOCHS = cfg.params.epochs_gpu


class TrainPipe(ABC):
    def __init__(self, wandb_enable):
        # self.use_amp = None
        self.read_dataset()
        self.wandb_enable = wandb_enable

    @abstractmethod
    def read_dataset(self):
        pass

    # # @staticmethod
    def _concat_index(self, df1, df2):
        df2.index = df2.index // cfg.params.neg_train
        return pd.concat([df1, df2], axis=0).sort_index(kind='mregesort').reset_index(drop=True)

    # @classmethod
    def _generate_all(self):
        self.df_train_pos[DEFAULT_RATING_COL] = 1
        df_train_all = self._concat_index(self.df_train_pos, self.df_train_neg)
        df_train_all['flag'] = 1
        self.df_test_ori['flag'] = 0
        self.df_all = pd.concat([df_train_all, self.df_test_ori], axis=0).reset_index(drop=True)

    @abstractmethod
    def features_select(self):
        pass

    def _cat_encode(self, df_data, list_f, encoder):
        for f in list_f:
            df_data[f] = encoder.fit_transform(df_data[f].astype('category').cat.codes.values)
        return df_data

    def _embedding_dimension(self, df_all_encode, features_to_train, max_dim=50):
        embedding_size = []
        features_to_em = [i for i in features_to_train if i != DEFAULT_RATING_COL]
        for c in features_to_em:
            num_unique_values = int(df_all_encode[c].nunique())
            embed_dim = int(min(np.ceil(num_unique_values / 2), max_dim))
            embedding_size.append([num_unique_values, embed_dim])
        return embedding_size

    def encode_data(self, features_to_code, features_to_train, max_dim=50):
        encoder = preprocessing.LabelEncoder()
        df_all_encode = self._cat_encode(self.df_mix_merge, features_to_code, encoder)
        df_train = df_all_encode[self.df_all.flag == 1]
        df_test = df_all_encode[self.df_all.flag == 0]
        self.df_train = df_train[features_to_train]
        self.df_test = df_test[features_to_train]
        self.embedding_size = self._embedding_dimension(df_all_encode, features_to_train, max_dim)
        # self.df_train, self.df_test, embedding_size

    @abstractmethod
    def data_check(self):
        pass

    def df_to_np(self):
        '''
        Creat the numpy array for training
        '''
        df_train_split, df_val_split = data_split_user(self.df_train, val_size=0.2)

        np_train = df_train_split.values
        np_val = df_val_split.values
        np_test = self.df_test.values
        return np_train, np_val, np_test

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @abstractmethod
    def dataloader_init(self, np_train, np_val, np_test):
        pass

    @abstractmethod
    def wandb_init(self):
        pass

    def run(self):
        self.read_dataset()
        self._generate_all()
        self.features_select()
        self.df_mix_merge = mix_merge(self.df_all, self.df_all_features, self.user_features_extend,
                                      self.item_features_extend)
        num_feature = []
        features_to_code = self.df_mix_merge.columns
        features_to_train = [DEFAULT_USER_COL, DEFAULT_ITEM_COL] + self.user_features + self.item_features + [
            DEFAULT_RATING_COL]

        self.encode_data(features_to_code, features_to_train, max_dim=cfg.params.emb_dim)
        self.data_check()
        self.df_to_np()
        np_train, np_val, np_test = self.df_to_np()
        train_loader, val_loader, test_loader = self.dataloader_init(np_train, np_val, np_test)

        self.model = EntityCat(embedding_size=self.embedding_size, num_numerical_cols=len(num_feature),
                               output_size=1)
        self.model.to(device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.params.lr)

        def output_trans_loss(output):
            return output['y_pred'], output['label']

        val_metrics_train = {
            'auc': CustomAuc(),
            "loss": Loss(self.criterion, output_transform=output_trans_loss)
        }

        val_metrics_test = {
            'hr': CustomHR(k=cfg.params.topk),
            'ndcg': CustomNDCG(k=cfg.params.topk),
            'auc': CustomAuc(),
            'auc_top': CustomAuc_top(),
            'recall_top': CustomRecall_top(k=cfg.params.topk),
            'precision_top': CustomPrecision_top(k=cfg.params.topk),
            "loss": Loss(self.criterion, output_transform=output_trans_loss)
        }

        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        def train_step(engine, batch):
            self.model.train()
            self.optimizer.zero_grad()
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                y_pred = self.model(x).reshape(1, -1).flatten()
                loss = self.criterion(y_pred, y.float())
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            return loss.item()

        def validation_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, label = batch[0].to(device), batch[1].to(device)
                y_pred = self.model(x).reshape(1, -1).flatten()
                label = label.float()
                return {'label': label, 'y_pred': y_pred}

        def test_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, label = batch[0].to(device), batch[1].to(device)
                y_pred = self.model(x).reshape(1, -1).flatten()
                label = label.float()
                y_pred_top, indices = torch.topk(y_pred, engine.state.topk)
                y_pred_top = y_pred_top.detach().cpu().numpy()
                reco_item = torch.take(x[:, 1], indices).cpu().numpy().tolist()
                pos_item = x[0, 1].cpu().numpy().tolist()  # ground truth, item id
                label_top = label[indices].cpu().numpy()
                indices = indices.cpu().numpy()
                return {'pos_item': pos_item, 'reco_item': reco_item, 'y_pred_top': y_pred_top,
                        'label_top': label_top, 'label': label, 'y_pred': y_pred, 'y_indices': indices}

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
            test_evaluator.state.topk = cfg.params.topk

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

        Eearly_stop_handler = EarlyStopping(patience=cfg.params.patience, score_function=score_function,
                                            trainer=trainer)

        # val_evaluator.add_event_handler(Events.COMPLETED, Eearly_stop_handler)

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
            print(f"Test Results - Epoch[{trainer.state.epoch}]  Avg loss: {loss:.2f}   \
                   Avg ndcg: {ndcg:.2f}  Avg auc: {auc:.2f}  Avg auc_top: {auc_top:.2f}     \
                     Avg recall: {recall:.2f}  Avg precision: {precision:.2f} Avg hr: {hr:.2f}")

        pbar = ProgressBar(persist=False)
        pbar.attach(trainer)
        # trainer.run(train_loader, max_epochs=2)
        if self.wandb_enable:
            self.wandb_init()

            to_save = {'model': self.model}
            checkpoint_handler = ModelCheckpoint(
                self.wandb_logger.run.dir,
                n_saved=1, filename_prefix='best',
                score_name="auc",
                global_step_transform=global_step_from_engine(trainer)
            )

            val_evaluator.add_event_handler(Events.COMPLETED, checkpoint_handler, to_save)

            self.wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.ITERATION_COMPLETED,
                tag="training",
                output_transform=lambda loss: {"loss": loss}
            )

            self.wandb_logger.attach_output_handler(
                train_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="training",
                metric_names=['loss', 'auc'],
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            self.wandb_logger.attach_output_handler(
                val_evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="validation",
                metric_names=['loss', "auc"],
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            self.wandb_logger.attach_output_handler(
                test_evaluator,
                event_name=Events.COMPLETED,
                tag="test",
                metric_names=['loss', "auc", 'hr', 'ndcg', 'auc_top', 'recall_top', 'precision_top'],
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            self.wandb_logger.attach_opt_params_handler(
                trainer,
                event_name=Events.ITERATION_STARTED,
                optimizer=self.optimizer,
                param_name='lr'  # optional
            )

            # wandb_logger.watch(model)
            trainer.run(train_loader, max_epochs=EPOCHS)
            self.wandb_logger.close()
        else:
            trainer.run(train_loader, max_epochs=2)


class model_leave_one(TrainPipe):
    def __init__(self, wandb_enable: bool):
        super().__init__(wandb_enable=wandb_enable)
        # self.wanab_enable = wandb_enable

    def read_dataset(self):
        if device.type == 'cpu':
            self.use_amp = False
            self.df_train_pos = ng_sample.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_pos))
            self.df_train_neg = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_neg))
            self.df_test_ori = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.test_pos_neg)).iloc[:202, ]
            self.df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
            self.df_train_pos = self.df_train_pos.sort_values(by=[DEFAULT_USER_COL]).iloc[:100, ].reset_index(drop=True)
            self.df_train_neg = self.df_train_neg.sort_values(by=[DEFAULT_USER_COL]).iloc[:100 * cfg.params.neg_train, ].reset_index(drop=True)
        else:
            self.use_amp = True
            self.df_train_pos = ng_sample.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_pos))
            self.df_train_neg = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_neg))
            self.df_test_ori = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.test_pos_neg))
            self.df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
            self.df_train_pos = self.df_train_pos.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)
            self.df_train_neg = self.df_train_neg.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)

    def features_select(self):
        self.user_features = ['DegreeType', 'Major', 'GraduationDate']
        self.user_features_extend = [DEFAULT_USER_COL] + self.user_features

        self.item_features = []
        self.item_features_extend = [DEFAULT_ITEM_COL] + self.item_features

        self.base_features = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        # return user_features_extend, item_features_extend, base_features

    def data_check(self):
        assert len(self.df_train[self.df_train.rating == 0]) / len(
            self.df_train[self.df_train.rating == 1]) == cfg.params.neg_train, 'wrong neg/pos ratio in training set'
        assert len(self.df_test[self.df_test.rating == 0]) / len(
            self.df_test[self.df_test.rating == 1]) == cfg.params.neg_test, 'wrong neg/pos ratio in test set '
        # Check if all the users in test can be found in training set
        assert sum(np.isin(self.df_test.userid.unique(), self.df_train.userid.unique(), assume_unique=True)) == len(
            self.df_test.userid.unique()), 'cold start'
        # The the uniqueness of items between training and test. For a user, on common items between training and test dataset.
        assert self.df_all.shape[0] == self.df_train.shape[0] + self.df_test.shape[0], 'wrong data concat'
        assert sum(self.df_all.groupby(['userid']).apply(lambda x: len(x['itemid'].unique()))) == self.df_all.shape[
            0], 'train and test have overlap item'

    def dataloader_init(self, np_train, np_val, np_test):
        g = torch.Generator()
        g.manual_seed(0)

        train_dataset = CatData(np_train)
        val_dataset = CatData(np_val)
        test_dataset = CatData(np_test)
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                       worker_init_fn=self.seed_worker, generator=g)
        val_loader = data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0,
                                     worker_init_fn=self.seed_worker, generator=g)
        test_loader = data.DataLoader(test_dataset, batch_size=cfg.params.neg_test + 1, shuffle=False, num_workers=0,
                                      worker_init_fn=self.seed_worker, generator=g)
        return train_loader, val_loader, test_loader

    def wandb_init(self):
        config_dict = dict(cfg.params)
        config_dict['Features'] = '-'.join(self.user_features + self.item_features)
        self.wandb_logger = WandBLogger(
            project="pytorch-jrs",
            name="-".join(self.user_features) + '-' + '-'.join(self.item_features),
            config=config_dict,
            tags=['leave_one']
        )


class model_temp(TrainPipe):
    def __init__(self, wandb_enable: bool):
        super().__init__(wandb_enable=wandb_enable)

    def read_dataset(self):
        if device.type == 'cpu':
            self.use_amp = False
            self.df_train_pos = ng_sample.read_feather(pathlib.Path(cfg.path.global_temp, cfg.global_temp_data.train_pos))
            self.df_train_neg = pd.read_feather(pathlib.Path(cfg.path.global_temp, cfg.global_temp_data.train_neg))
            self.df_test_ori = pd.read_feather(pathlib.Path(cfg.path.global_temp, cfg.global_temp_data.test_neg)).iloc[:100, ]
            self.df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
            self.df_train_pos = self.df_train_pos.sort_values(by=[DEFAULT_USER_COL]).iloc[:100, ].reset_index(drop=True)
            self.df_train_neg = self.df_train_neg.sort_values(by=[DEFAULT_USER_COL]).iloc[:100 * cfg.params.neg_train, ].reset_index(drop=True)
        else:
            self.use_amp = True
            self.df_train_pos = ng_sample.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_pos))
            self.df_train_neg = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_neg))
            self.df_test_ori = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.test_pos_neg))
            self.df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
            self.df_train_pos = self.df_train_pos.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)
            self.df_train_neg = self.df_train_neg.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)

    def features_select(self):
        self.user_features = ['DegreeType', 'Major', 'GraduationDate']
        self.user_features_extend = [DEFAULT_USER_COL] + self.user_features

        self.item_features = []
        self.item_features_extend = [DEFAULT_ITEM_COL] + self.item_features

        self.base_features = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        # return user_features_extend, item_features_extend, base_features

    def data_check(self):
        assert len(self.df_train[self.df_train.rating == 0]) / len(
            self.df_train[self.df_train.rating == 1]) == cfg.params.neg_train, 'wrong neg/pos ratio in training set'
        # Check if all the users in test can be found in training set
        assert sum(np.isin(self.df_test.userid.unique(), self.df_train.userid.unique(), assume_unique=True)) == len(
            self.df_test.userid.unique()), 'cold start'
        # The the uniqueness of items between training and test. For a user, on common items between training and test dataset.
        assert self.df_all.shape[0] == self.df_train.shape[0] + self.df_test.shape[0], 'wrong data concat'
        assert sum(self.df_all.groupby(['userid']).apply(lambda x: len(x['itemid'].unique()))) == self.df_all.shape[
            0], 'train and test have overlap item'

    def dataloader_init(self, np_train, np_val, np_test):
        g = torch.Generator()
        g.manual_seed(0)

        train_dataset = CatData(np_train)
        val_dataset = CatData(np_val)
        test_dataset = CatData(np_test)
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                       worker_init_fn=self.seed_worker, generator=g)
        val_loader = data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0,
                                     worker_init_fn=self.seed_worker, generator=g)
        test_loader = data.DataLoader(test_dataset, batch_size=cfg.params.neg_test, shuffle=False, num_workers=0,
                                      worker_init_fn=self.seed_worker, generator=g)
        return train_loader, val_loader, test_loader

    def wandb_init(self):
        config_dict = dict(cfg.params)
        config_dict['Features'] = '-'.join(self.user_features + self.item_features)
        self.wandb_logger = WandBLogger(
            project="pytorch-jrs",
            name="-".join(self.user_features) + '-' + '-'.join(self.item_features),
            config=config_dict,
            tags=['temp']
        )


class model_sbert():
    def __init__(self, wandb_enable: bool):
        super().__init__(wandb_enable=wandb_enable)
        # self.wanab_enable = wandb_enable

    def read_dataset(self):
        if device.type == 'cpu':
            self.use_amp = False
            self.df_train_pos = ng_sample.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_pos))
            self.df_train_neg = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_neg))
            self.df_test_ori = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.test_pos_neg)).iloc[
                               :202, ]
            self.df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
            self.df_train_pos = self.df_train_pos.sort_values(by=[DEFAULT_USER_COL]).iloc[:100, ].reset_index(drop=True)
            self.df_train_neg = self.df_train_neg.sort_values(by=[DEFAULT_USER_COL]).iloc[
                                :100 * cfg.params.neg_train, ].reset_index(drop=True)
        else:
            self.use_amp = True
            self.df_train_pos = ng_sample.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_pos))
            self.df_train_neg = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.train_neg))
            self.df_test_ori = pd.read_feather(pathlib.Path(cfg.path.leave_one, cfg.leave_one_data.test_pos_neg))
            self.df_all_features = pd.read_csv(pathlib.Path(cfg.path.root, cfg.data.all_features))
            self.df_train_pos = self.df_train_pos.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)
            self.df_train_neg = self.df_train_neg.sort_values(by=[DEFAULT_USER_COL]).reset_index(drop=True)

    def features_select(self):
        self.user_features = ['DegreeType', 'Major', 'GraduationDate']
        self.user_features_extend = [DEFAULT_USER_COL] + self.user_features

        self.item_features = []
        self.item_features_extend = [DEFAULT_ITEM_COL] + self.item_features

        self.base_features = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]
        # return user_features_extend, item_features_extend, base_features

    def data_check(self):
        assert len(self.df_train[self.df_train.rating == 0]) / len(
            self.df_train[self.df_train.rating == 1]) == cfg.params.neg_train, 'wrong neg/pos ratio in training set'
        assert len(self.df_test[self.df_test.rating == 0]) / len(
            self.df_test[self.df_test.rating == 1]) == cfg.params.neg_test, 'wrong neg/pos ratio in test set '
        # Check if all the users in test can be found in training set
        assert sum(np.isin(self.df_test.userid.unique(), self.df_train.userid.unique(), assume_unique=True)) == len(
            self.df_test.userid.unique()), 'cold start'
        # The the uniqueness of items between training and test. For a user, on common items between training and test dataset.
        assert self.df_all.shape[0] == self.df_train.shape[0] + self.df_test.shape[0], 'wrong data concat'
        assert sum(self.df_all.groupby(['userid']).apply(lambda x: len(x['itemid'].unique()))) == self.df_all.shape[
            0], 'train and test have overlap item'

    def dataloader_init(self, np_train, np_val, np_test):
        g = torch.Generator()
        g.manual_seed(0)

        train_dataset = CatData(np_train)
        val_dataset = CatData(np_val)
        test_dataset = CatData(np_test)
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                                       worker_init_fn=self.seed_worker, generator=g)
        val_loader = data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0,
                                     worker_init_fn=self.seed_worker, generator=g)
        test_loader = data.DataLoader(test_dataset, batch_size=cfg.params.neg_test + 1, shuffle=False, num_workers=0,
                                      worker_init_fn=self.seed_worker, generator=g)
        return train_loader, val_loader, test_loader

    def wandb_init(self):
        config_dict = dict(cfg.params)
        config_dict['Features'] = '-'.join(self.user_features + self.item_features)
        self.wandb_logger = WandBLogger(
            project="pytorch-jrs",
            name="-".join(self.user_features) + '-' + '-'.join(self.item_features),
            config=config_dict,
            tags=['leave_one']
        )


if __name__ == "__main__":
    train = model_temp(wandb_enable=True)
    train.run()
