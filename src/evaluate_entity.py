import numpy as np
import torch
from sklearn.metrics import roc_auc_score

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
    return roc_auc_score(gt_item, pred_prob)


#
# def metrics(model, test_loader, top_k):
# 	HR, NDCG, ROC = [], [], []
#
# 	for cat_data, label in test_loader:
# 		cat_data = cat_data.to(device)
# 		label = label.cpu().numpy().tolist()
#
# 		predictions = model(cat_data)[:, 1]
# 		prob, indices = torch.topk(predictions, top_k)
# 		recommends = torch.take(
# 				cat_data[:, 1], indices).cpu().numpy().tolist()
#
# 		gt_item = cat_data[0, 1].cpu().numpy().tolist() #ground truth, item id
# 		pred_list = predictions.tolist()
# 		HR.append(hit(gt_item, recommends))
# 		NDCG.append(ndcg(gt_item, recommends))
# 		ROC.append(roc(label, pred_list))
#
# 	return np.mean(HR), np.mean(NDCG), np.mean(ROC)


def metrics(cat_data, predictions, label, top_k, is_train):
    if is_train:
        HR_1batch, NDCG_1batch, ROC_1batch =0,0,0
    else:
        prob, indices = torch.topk(predictions, top_k)
        recommends = torch.take(
            cat_data[:, 1], indices).cpu().numpy().tolist()
        gt_item = cat_data[0, 1].cpu().numpy().tolist()  # ground truth, item id
        pred_list = predictions.tolist()
        HR_1batch = hit(gt_item, recommends)
        NDCG_1batch = ndcg(gt_item, recommends)
        ROC_1batch = roc(label.cpu().numpy().tolist(), pred_list)

    return HR_1batch, NDCG_1batch, ROC_1batch