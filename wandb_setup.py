import wandb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from unet import *
from dataset import *
from transform import train_transform, val_transform
from unet_b4 import *
from unet_smp import *

def wandb_train():
    with wandb.init() as run:
        config = wandb.config

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        # Load datasets
        train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
        val_dataset = CustomDataset(X_val, y_val, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, pin_memory=True)

        #-----------------------------------------------

        #Uncomment the model you want to use while commenting others to use your desired type of model.

        # #Using normal Unet from scratch (uncomment to use this)
        # model = UNet().to(device)

        # #Using Unet from scratch with efficientb4 as encoder (uncomment to use this)
        # model = UNetEfficientNetB4().to(device)

        # Using Pre-exisitng Unet model from smp (uncomment to use this)
        model = model_smp.to(device)

        #------------------------------------------------

        # Optimizer
        if config.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")

        # Loss function
        criterion = smp.losses.DiceLoss(mode="binary")

        # train loop
        for epoch in range(15):
            model.train()
            total_loss = 0.0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_features)
                outputs = torch.sigmoid(outputs)

                # Ensuring shape match
                if outputs.shape != batch_labels.shape:
                    outputs = torch.nn.functional.interpolate(outputs, size=batch_labels.shape[-2:], mode='bilinear', align_corners=False)

                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_features)
                    outputs = torch.sigmoid(outputs)

                    if outputs.shape != batch_labels.shape:
                        outputs = torch.nn.functional.interpolate(outputs, size=batch_labels.shape[-2:], mode='bilinear', align_corners=False)

                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
            })


def start_sweep(sweep_id, count=5):
    wandb.agent(sweep_id, function=wandb_train, count=count)
