import torch
import torch.nn.functional as F


def confusion(pred, target, num_classes):
    k = target * num_classes + pred
    return torch.bincount(k.flatten(), minlength=num_classes ** 2).view(num_classes, num_classes)


def scores_from_confusion(cm):
    cm = cm.double()
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp(min=1)
    dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    pixel_acc = tp.sum() / cm.sum()
    return {
        "iou": iou.tolist(), "dice": dice.tolist(),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "miou": iou.mean().item(), "mdice": dice.mean().item(),
        "pixel_acc": pixel_acc.item(),
    }


def dice_loss(logits, target, eps=1.0):
    num_classes = logits.shape[1]
    prob = logits.softmax(1)
    onehot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    inter = (prob * onehot).sum((0, 2, 3))
    union = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def seg_loss(logits, target):
    return F.cross_entropy(logits, target) + dice_loss(logits, target)
