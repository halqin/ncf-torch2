import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.metrics import  Metric
from metrics import ranking
import hydra
from omegaconf import DictConfig

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# device = 'cpu'
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
class CustomAuc(Metric):
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
        pd_scores = output[1].numpy()
        self._auc_list.append(self.auc.compute(pd_scores = pd_scores, gt_pos =y))

    @sync_all_reduce("_auc_list")
    def compute(self):
        return np.mean(self._auc_list)


# def process_function()
class CustomAuc_top(Metric):
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
        self._auc_list.append(self.auc.compute(pd_scores = pd_scores, gt_pos =y))

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