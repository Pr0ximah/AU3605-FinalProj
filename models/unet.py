from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(UNet, self).__init__()
        # 512x512
        self.conv1 = ConvBlock(in_channels, 64)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        # 256x256
        self.conv2 = ConvBlock(64, 128)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        # 128x128
        self.conv3 = ConvBlock(128, 256)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        # 64x64
        self.conv4 = ConvBlock(256, 512)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        # 32x32
        self.conv5 = ConvBlock(512, 1024)
        # 32x32
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = ConvBlock(1024, 512)
        # 64x64
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = ConvBlock(512, 256)
        # 128x128
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = ConvBlock(256, 128)
        # 256x256
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = ConvBlock(128, 64)
        # 512x512
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool1(x1))
        x3 = self.conv3(self.maxpool2(x2))
        x4 = self.conv4(self.maxpool3(x3))
        x5 = self.conv5(self.maxpool4(x4))
        upconv1 = self.upconv1(x5)
        x6 = self.conv6(torch.cat((x4, upconv1), dim=1))
        upconv2 = self.upconv2(x6)
        x7 = self.conv7(torch.cat((x3, upconv2), dim=1))
        upconv3 = self.upconv3(x7)
        x8 = self.conv8(torch.cat((x2, upconv3), dim=1))
        upconv4 = self.upconv4(x8)
        x9 = self.conv9(torch.cat((x1, upconv4), dim=1))
        output = self.output(x9)
        output = torch.sigmoid(output)
        return output
