import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from transform import val_transform

def visualize_predictions(model, device, X_test, y_test, num_samples_to_plot=5):
    model.eval()
    test_dataset = CustomDataset(X_test, y_test, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)

    count = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            outputs = torch.sigmoid(outputs)

            binary_preds = (outputs > 0.5).float()
            binary_labels = (batch_labels > 0.5).float()

            for i in range(batch_features.shape[0]):
                if count >= num_samples_to_plot:
                    break

                image = batch_features[i].cpu().permute(1, 2, 0).numpy()
                gt_mask = binary_labels[i].cpu().squeeze().numpy()
                pred_mask = binary_preds[i].cpu().squeeze().numpy()

                plt.figure(figsize=(12, 4))

                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask, cmap='gray')
                plt.title("Ground Truth Mask")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title("Predicted Mask")
                plt.axis('off')

                plt.tight_layout()
                plt.show()

                count += 1

            if count >= num_samples_to_plot:
                break
