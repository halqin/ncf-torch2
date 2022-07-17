import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.metrics import Metric
from src.metrics import ranking
from src.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL

import hydra
from omegaconf import DictConfig

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def auc(gt_item, pred_prob):
    try:
        return roc_auc_score(gt_item, pred_prob)
    except:
        return 0


def recall(gt, prob, th):
    y_pred = [1 if y >= th else 0 for y in prob]
    return recall_score(gt, y_pred, zero_division=0)


def precision(gt, prob, th):
    y_pred = [1 if y >= th else 0 for y in prob]
    return precision_score(gt, y_pred, zero_division=0)


#
# def map(gt_item, pred_prob):
#
#     return

def metrics(cat_data, predictions, label, top_k, threshold=0.5):
    # if is_train:
    #     HR_1batch, NDCG_1batch, ROC_1batch, ROC_top1batch, recall_1batch,  precision_1batch =0,0,0,0,0,0
    # else:
    '''
    cat_data: the inpout data
    '''
    y_score, indices = torch.topk(predictions, top_k)
    recommends = torch.take(
        cat_data[:, 1], indices).cpu().numpy().tolist()
    gt_item = cat_data[0, 1].cpu().numpy().tolist()  # ground truth, item id
    y_score_all = predictions.tolist()
    HR_1batch = hit(gt_item, recommends)
    NDCG_1batch = ndcg(gt_item, recommends)
    auc_1batch = auc(label.cpu().numpy(), y_score_all)
    auc_top1batch = auc(label[indices].cpu().numpy(), y_score.detach().cpu().numpy())
    recall_1batch = recall(label[indices].cpu().numpy(), y_score.detach().cpu().numpy(), threshold)
    precision_1batch = precision(label[indices].cpu().numpy(), y_score.detach().cpu().numpy(), threshold)
    return HR_1batch, NDCG_1batch, auc_1batch, auc_top1batch, recall_1batch, precision_1batch


# def process_function()
class CustomHR(Metric):
    '''
    Calcualte Hit Rate
    '''

    def __init__(self, output_transform=lambda x: [x['pos_item'], x['reco_item']], device="cpu"):
        self._hit_list = None
        super(CustomHR, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._hit_list = []
        super(CustomHR, self).reset()

    @reinit__is_reduced
    def update(self, output):
        gt_item = output[0]
        reco_item = output[1]
        self._hit_list.append(hit(gt_item, reco_item))

    @sync_all_reduce("_hit_list")
    def compute(self):
        return np.mean(self._hit_list)


# def process_function()
class CustomNDCG(Metric):

    def __init__(self, k, output_transform=lambda x: [x['label'], x['y_indices']], device="cpu"):
        self._ndcg_list = None
        self.k = k
        self.ndcg = ranking.NDCG(k=k)
        super(CustomNDCG, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._ndcg_list = []
        super(CustomNDCG, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y = output[0].cpu().numpy()
        y_indices = output[1]
        self._ndcg_list.append(self.ndcg.compute(y, y_indices))

    @sync_all_reduce("_ndcg_list")
    def compute(self):
        return np.mean(self._ndcg_list)


# def process_function()
class CustomAuc_new(Metric):
    '''
    Calcualte AUC
    '''

    def __init__(self, output_transform=lambda x: [x['label'], x['y_pred']], device="cpu"):
        self._auc_rate = None
        self.auc = ranking.AUC()
        super(CustomAuc, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._auc_list = []
        super(CustomAuc, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y = output[0].cpu().numpy().astype(int)
        pd_scores = output[1].cpu().numpy()
        self._auc_list.append(self.auc.compute(pd_scores=pd_scores, gt_pos=y))

    @sync_all_reduce("_auc_list")
    def compute(self):
        return np.mean(self._auc_list)


# def process_function()
class CustomAuc_top_new(Metric):
    '''
    Calcualte AUC@k
    '''

    def __init__(self, output_transform=lambda x: [x['label_top'], x['y_pred_top']], device="cpu"):
        self._auc_list = None
        self.auc = ranking.AUC()
        super(CustomAuc_top, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._auc_list = []
        super(CustomAuc_top, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y = output[0].astype(int)
        pd_scores = output[1]
        self._auc_list.append(self.auc.compute(pd_scores=pd_scores, gt_pos=y))

    @sync_all_reduce("_auc_list")
    def compute(self):
        return np.mean(self._auc_list)


class CustomRecall_top(Metric):
    '''
    Calcualte Recall
    '''

    # @hydra.main(version_base=None, config_path='../conf', config_name='config')
    def __init__(self, k, output_transform=lambda x: [x['label'], x['y_indices']], device="cpu"):
        self._recall_list_list = None
        self._metric_r = ranking.Recall(k=k)
        super(CustomRecall_top, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._recall_list = []
        super(CustomRecall_top, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y = output[0].cpu().numpy()
        y_indices = output[1]
        self._recall_list.append(self._metric_r.compute(y, y_indices))

    @sync_all_reduce("_recall_list")
    def compute(self):
        return np.mean(self._recall_list)


class CustomPrecision_top(Metric):
    '''
    Calcualte Hit Rate
    '''

    # @hydra.main(version_base=None, config_path='../conf', config_name='config')
    def __init__(self, k, output_transform=lambda x: [x['label'], x['y_indices']], device="cpu"):
        self._precision_list = None
        self._metric_p = ranking.Precision(k=k)
        super(CustomPrecision_top, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._precision_list = []
        super(CustomPrecision_top, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y = output[0].cpu().numpy()
        y_indices = output[1]
        self._precision_list.append(self._metric_p.compute(y, y_indices))

    @sync_all_reduce("_precision_list")
    def compute(self):
        return np.mean(self._precision_list)


class CustomAuc(Metric):
    '''
    Calcualte Auc
    '''

    def __init__(self, output_transform=lambda x: [x['label'], x['y_pred']], device="cpu"):
        self._roc_rate = None
        super(CustomAuc, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._roc_list = []
        super(CustomAuc, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y = output[0].cpu().numpy()
        y_pred = output[1].tolist()
        self._roc_list.append(auc(y, y_pred))

    @sync_all_reduce("_roc_list")
    def compute(self):
        return np.mean(self._roc_list)


class CustomAuc_top(Metric):
    '''
    Calcualte Auc@k
    '''

    def __init__(self, output_transform=lambda x: [x['label_top'], x['y_pred_top']], device="cpu"):
        self._roc_list = None
        super(CustomAuc_top, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._roc_list = []
        super(CustomAuc_top, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y = output[0]
        y_pred = output[1]
        self._roc_list.append(auc(y, y_pred))

    @sync_all_reduce("_roc_list")
    def compute(self):
        return np.mean(self._roc_list)


def model_infer2(df_true, jobsid, usersid, model, u_i_matrix, n):
    test_items_indices, test_items = indices_extract(df_true, jobsid, DEFAULT_ITEM_COL)
    test_users_indices, test_users = indices_extract(df_true, usersid, DEFAULT_USER_COL)
    test_items_rating = df_true[DEFAULT_RATING_COL].values

    ids, scores = model.recommend(test_users_indices,
                                  u_i_matrix[test_users_indices],
                                  N=n,
                                  items=test_items_indices)
    # reco_jobsid = jobsid[ids][0]
    reco_indices, test_items_rating = indices_search(test_items, jobsid, test_items_rating, ids)
    return test_items_rating.flatten(), reco_indices.flatten(), scores.flatten()


def indices_search(items, jobsid, rating, ids):
    reco_jobsid = jobsid[ids]

    sort_ind = np.argsort(items)
    sort_rating, sort_item = rating[sort_ind], items[sort_ind]

    sort_item_filter = sort_item[np.isin(sort_item,reco_jobsid)]
    sort_rating_filter = sort_rating[np.isin(sort_item, reco_jobsid)]

    reco_indices = np.searchsorted(sort_item_filter, reco_jobsid)
    return reco_indices, sort_rating_filter


def indices_extract(df, x_list, feature):
    val = df[feature].values
    indices = np.where(np.isin(x_list, val))[0]
    return indices, val
