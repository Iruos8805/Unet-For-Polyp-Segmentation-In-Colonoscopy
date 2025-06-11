def compute_iou(preds, targets, eps=1e-7):
    preds = preds.float()
    targets = targets.float()

    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def compute_dice(preds, targets, eps=1e-7):
    preds = preds.float()
    targets = targets.float()

    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    dice = (2. * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    return dice.mean().item()
