import torch
import torch.nn as nn

def d_conv(in_channels, out_channels):
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op

class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        # Contracting path
        self.down_dconv1 = d_conv(3, 64)
        self.down_dconv2 = d_conv(64, 128)
        self.down_dconv3 = d_conv(128, 256)
        self.down_dconv4 = d_conv(256, 512)
        self.down_dconv5 = d_conv(512, 1024)

        # Expanding path
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_dconv1 = d_conv(1024, 512)

        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_dconv2 = d_conv(512, 256)

        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_dconv3 = d_conv(256, 128)

        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_dconv4 = d_conv(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Downsampling + max pooling
        down_1 = self.down_dconv1(x)
        down_max_1 = self.max_pool2d(down_1)

        down_2 = self.down_dconv2(down_max_1)
        down_max_2 = self.max_pool2d(down_2)

        down_3 = self.down_dconv3(down_max_2)
        down_max_3 = self.max_pool2d(down_3)

        down_4 = self.down_dconv4(down_max_3)
        down_max_4 = self.max_pool2d(down_4)

        down_5 = self.down_dconv5(down_max_4)

        # Upsampling + concatenation
        up_1 = self.up_conv1(down_5)
        x = self.up_dconv1(torch.cat([down_4, up_1], dim=1))

        up_2 = self.up_conv2(x)
        x = self.up_dconv2(torch.cat([down_3, up_2], dim=1))

        up_3 = self.up_conv3(x)
        x = self.up_dconv3(torch.cat([down_2, up_3], dim=1))

        up_4 = self.up_conv4(x)
        x = self.up_dconv4(torch.cat([down_1, up_4], dim=1))

        out = self.out(x)
        return out
