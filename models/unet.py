from torch import nn
import torch


class UNet(nn.Module):
    # input size: 572x572
    def __init__(self, in_channels, out_channels=1):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
        )
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
        )
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=True),
        )
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
        )
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 512, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=True),
        )
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=True),
        )
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
        )
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Conv2d(64, out_channels, 1)
        self.dropout = nn.Dropout(0.2)

    def center_crop(self, x, target_tensor):
        _, _, tH, tW = target_tensor.size()
        _, _, xH, xW = x.size()
        startx = xW // 2 - tW // 2
        starty = xH // 2 - tH // 2
        return x[:, :, starty : starty + tH, startx : startx + tW]

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool1(x1))
        x3 = self.conv3(self.maxpool2(x2))
        x4 = self.conv4(self.maxpool3(x3))
        x5 = self.conv5(self.maxpool4(x4))
        upconv1 = self.upconv1(x5)
        x6 = self.conv6(torch.cat((self.center_crop(x4, upconv1), upconv1), dim=1))
        upconv2 = self.upconv2(x6)
        x7 = self.conv7(torch.cat((self.center_crop(x3, upconv2), upconv2), dim=1))
        upconv3 = self.upconv3(x7)
        x8 = self.conv8(torch.cat((self.center_crop(x2, upconv3), upconv3), dim=1))
        upconv4 = self.upconv4(x8)
        x9 = self.conv9(torch.cat((self.center_crop(x1, upconv4), upconv4), dim=1))
        output = self.output(x9)
        output = torch.sigmoid(output)
        return output
