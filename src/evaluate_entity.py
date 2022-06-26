import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

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


def roc(gt_item, pred_prob):
    try:
        return roc_auc_score(gt_item, pred_prob)
    except:
        pass

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

def metrics(cat_data, predictions, label, top_k, is_train, threshold=0.5):
    if is_train:
        HR_1batch, NDCG_1batch, ROC_1batch, ROC_top1batch, recall_1batch,  precision_1batch =0,0,0,0,0,0
    else:
        y_score, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            cat_data[:, 1], indices).cpu().numpy().tolist()
        gt_item = cat_data[0, 1].cpu().numpy().tolist()  # ground truth, item id
        pred_list = predictions.tolist()
        HR_1batch = hit(gt_item, recommends)
        NDCG_1batch = ndcg(gt_item, recommends)
        ROC_1batch = roc(label.cpu().numpy(), pred_list)
        ROC_top1batch = roc(label[indices].cpu().numpy(), y_score.detach().cpu().numpy())
        recall_1batch = recall(label[indices].cpu().cpu().numpy(), y_score.detach().cpu().numpy(), threshold)
        precision_1batch = precision(label[indices].cpu().numpy(), y_score.detach().cpu().numpy(), threshold)
    return HR_1batch, NDCG_1batch, ROC_1batch, ROC_top1batch, recall_1batch, precision_1batch