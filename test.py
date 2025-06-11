import torch
import torch.nn.functional as F
from metrics import *
from transform import val_transform
from dataset import CustomDataset
from torch.utils.data import DataLoader

def test(model, device, X_test, y_test):
    model.eval()

    test_dataset = CustomDataset(X_test, y_test, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, pin_memory=True)

    ious, dices = [], []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Ensuring batch_labels shape is [B, 1, H, W]
            if batch_labels.dim() == 3:
                batch_labels = batch_labels.unsqueeze(1)

            outputs = torch.sigmoid(model(batch_features))

            # Resizing outputs to match batch_labels if needed
            if outputs.shape != batch_labels.shape:
                outputs = F.interpolate(outputs, size=batch_labels.shape[-2:], mode='bilinear', align_corners=False)

            binary_preds = (outputs > 0.5).float()
            binary_labels = (batch_labels > 0.5).float()

            ious.append(compute_iou(binary_preds.cpu(), binary_labels.cpu()))
            dices.append(compute_dice(binary_preds.cpu(), binary_labels.cpu()))

    avg_iou = sum(ious) / len(ious)
    avg_dice = sum(dices) / len(dices)

    print(f"Average IoU on test set: {avg_iou:.4f}")
    print(f"Average Dice Score on test set: {avg_dice:.4f}")

    return avg_iou, avg_dice
