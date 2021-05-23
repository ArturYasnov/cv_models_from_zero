# ====================================================
# Model
# ====================================================

class BaseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, out_pool):
        super().__init__()
        
        self.branch1 = BaseConvBlock(in_channels, out_1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            BaseConvBlock(in_channels, in_3x3, kernel_size=1),
            BaseConvBlock(in_channels, out_3x3, kernel_size=3, padding=1),
        )
        
        self.branch3 = nn.Sequential(
            BaseConvBlock(in_channels, in_5x5, kernel_size=1),
            BaseConvBlock(in_channels, out_5x5, kernel_size=5, padding=2),
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BaseConvBlock(in_channels, out_pool, kernel_size=1),
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        out = torch.cat([out1, out2, out3, out4], 1)
        return out


class InceptionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.opening = nn.Sequential(
            BaseConvBlock(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            BaseConvBlock(in_channels=64, out_channels=192, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # In this order: in_channels, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, out_pool
        self.Inceptions_part = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.opening(x)
        x = self.Inceptions_part(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout()
        x = self.fc(x)
        return out
