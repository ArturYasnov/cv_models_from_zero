# ====================================================
# Model
# ====================================================

class BaseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # print(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, block_mul, in_channels):
        super().__init__()

        self.conv1x1 = BaseConvBlock(in_channels, 64 * block_mul, kernel_size=1)

        self.block = nn.Sequential(
            BaseConvBlock(64 * block_mul, 64 * block_mul, kernel_size=3, padding=1),
            BaseConvBlock(64 * block_mul, 256 * block_mul, kernel_size=1),
        )

    def forward(self, x):
        x = self.conv1x1(x)
        return torch.cat([x, self.block(x)], 1)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = BaseConvBlock(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            ResidualBlock(block_mul=1, in_channels=64),
            ResidualBlock(block_mul=1, in_channels=320),
            ResidualBlock(block_mul=1, in_channels=320)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3_x = nn.Sequential(
            ResidualBlock(block_mul=2, in_channels=320),
            ResidualBlock(block_mul=2, in_channels=640),
            ResidualBlock(block_mul=2, in_channels=640),
            ResidualBlock(block_mul=2, in_channels=640)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4_x = nn.Sequential(
            ResidualBlock(block_mul=4, in_channels=640),
            ResidualBlock(block_mul=4, in_channels=1280),
            ResidualBlock(block_mul=4, in_channels=1280),
            ResidualBlock(block_mul=4, in_channels=1280),
            ResidualBlock(block_mul=4, in_channels=1280),
            ResidualBlock(block_mul=4, in_channels=1280)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv5_x = nn.Sequential(
            ResidualBlock(block_mul=8, in_channels=1280),
            ResidualBlock(block_mul=8, in_channels=2560),
            ResidualBlock(block_mul=8, in_channels=2560)
        )

        # 7*7*2560 = 125 440
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(2560, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2_x(x)
        x = self.pool2(x)
        x = self.conv3_x(x)
        x = self.pool3(x)
        x = self.conv4_x(x)
        x = self.pool4(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

