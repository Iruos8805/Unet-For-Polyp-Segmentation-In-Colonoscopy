import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from dataset import image_paths, mask_paths, CustomDataset
# from unet import UNet
from unet_b4 import *
from unet import *
from transform import *
import torch.nn.functional as F
from unet_smp import *

def train():

    # pre-train setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
    val_dataset = CustomDataset(X_val, y_val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, pin_memory=True)

    #-----------------------------------------------

    # #Using normal Unet from scratch
    # model = UNet().to(device)

    #Using Unet from scratch with efficientb4 as encoder
    model = UNetEfficientNetB4().to(device)

    # # Using Pre-exisitng Unet model from smp
    # model = model_smp.to(device)

    #------------------------------------------------

    # optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = smp.losses.DiceLoss(mode="binary")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        threshold=0.5,
        threshold_mode='abs',  # because threshold is absolute here
        verbose=True
    )

    train_dice_losses = []
    val_dice_losses = []

    for epoch in range(20):
        # train loop
        model.train()
        total_loss = 0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            batch_labels = batch_labels.unsqueeze(1)  # Shape: [B, 1, H, W]

            optimizer.zero_grad()
            outputs = torch.sigmoid(model(batch_features))

            # Fix spatial dimension mismatch
            if outputs.shape[-2:] != batch_labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=batch_labels.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_dice_losses.append(avg_loss)

        # val loop
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                batch_labels = batch_labels.unsqueeze(1)

                outputs = torch.sigmoid(model(batch_features))

                if outputs.shape[-2:] != batch_labels.shape[-2:]:
                    outputs = F.interpolate(outputs, size=batch_labels.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()


        avg_val_loss = val_loss / len(val_loader)
        val_dice_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model, device, X_test, y_test, train_dice_losses, val_dice_losses

