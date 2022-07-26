import torch






def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    with torch.cuda.amp.autocast(enabled=use_amp):
        y_pred = model(x).reshape(1, -1).flatten()
        loss = criterion(y_pred, y.float())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, label = batch[0].to(device), batch[1].to(device)
        y_pred = model(x).reshape(1, -1).flatten()
        label = label.float()
        return {'label': label, 'y_pred': y_pred}


def test_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, label = batch[0].to(device), batch[1].to(device)
        y_pred = model(x).reshape(1, -1).flatten()
        label = label.float()

        y_pred_top, indices = torch.topk(y_pred, engine.state.topk)

        y_pred_top = y_pred_top.detach().cpu().numpy()
        reco_item = torch.take(x[:, 1], indices).cpu().numpy().tolist()
        pos_item = x[0, 1].cpu().numpy().tolist()  # ground truth, item id
        label_top = label[indices].cpu().numpy()
        indices = indices.cpu().numpy()
        return {'pos_item': pos_item, 'reco_item': reco_item, 'y_pred_top': y_pred_top,
                'label_top': label_top, 'label': label, 'y_pred': y_pred, 'y_indices': indices}
