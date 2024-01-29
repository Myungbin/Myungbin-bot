import torch
import torch.nn as nn


def CELoss(pred_outs, labels):
    """
    pred_outs: [batch, clsNum]
    labels: [batch]
    """
    loss = nn.CrossEntropyLoss()
    try:
        loss_val = loss(pred_outs, labels)
    except:
        labels = torch.zeros(1).cuda().long()
        loss_val = loss(pred_outs.cuda(), labels)
    return loss_val
