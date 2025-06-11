import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

model_smp = smp.Unet(
    encoder_name= 'efficientnet-b4',
    encoder_weights= 'imagenet',
    in_channels=3,
    classes=1
)