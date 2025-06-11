import torch
import torch.nn as nn
import timm

def d_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNetEfficientNetB4(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Loading EfficientNet-B4 pretrained on ImageNet
        self.encoder = timm.create_model('efficientnet_b4', pretrained=True, features_only=True)

        # Output channels of EfficientNet-B4 feature stages
        enc_channels = self.encoder.feature_info.channels()  # [24, 32, 56, 160, 448]
        
        # Decoder layers
        self.up_conv1 = nn.ConvTranspose2d(enc_channels[4], enc_channels[3], kernel_size=2, stride=2)
        self.up_dconv1 = d_conv(enc_channels[3] + enc_channels[3], enc_channels[3])

        self.up_conv2 = nn.ConvTranspose2d(enc_channels[3], enc_channels[2], kernel_size=2, stride=2)
        self.up_dconv2 = d_conv(enc_channels[2] + enc_channels[2], enc_channels[2])

        self.up_conv3 = nn.ConvTranspose2d(enc_channels[2], enc_channels[1], kernel_size=2, stride=2)
        self.up_dconv3 = d_conv(enc_channels[1] + enc_channels[1], enc_channels[1])

        self.up_conv4 = nn.ConvTranspose2d(enc_channels[1], enc_channels[0], kernel_size=2, stride=2)
        self.up_dconv4 = d_conv(enc_channels[0] + enc_channels[0], enc_channels[0])

        self.out = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Extract encoder features
        feats = self.encoder(x)  # List of feature maps from 5 stages
        
        down1, down2, down3, down4, down5 = feats  # shallow to deep

        x = self.up_conv1(down5)
        x = self.up_dconv1(torch.cat([x, down4], dim=1))

        x = self.up_conv2(x)
        x = self.up_dconv2(torch.cat([x, down3], dim=1))

        x = self.up_conv3(x)
        x = self.up_dconv3(torch.cat([x, down2], dim=1))

        x = self.up_conv4(x)
        x = self.up_dconv4(torch.cat([x, down1], dim=1))

        return self.out(x)
