# ====================================================
# Model
# ====================================================

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Lauer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU()

        # Lauer 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Lauer 3
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu3 = nn.ReLU()

        # Lauer 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Lauer 5
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

        # Lauer 6
        self.conv6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()

        # Lauer 7
        self.conv7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        # Lauer 8
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2)

        # Lauer 9
        self.dropout1 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(9216, 4096)
        self.relu9 = nn.ReLU()

        # Lauer 10
        self.dropout2 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu10 = nn.ReLU()

        # Lauer 11
        self.linear3 = nn.Linear(4096, 1000)

        # self.bn1 = torch.nn.BatchNorm2d(16)
        # self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        # Lauer 1
        out = self.conv1(x)
        out = self.relu1(out)
        # Lauer 2
        out = self.maxpool2(out)
        # Lauer 3
        out = self.conv3(out)
        out = self.relu3(out)
        # Lauer 4
        out = self.maxpool4(out)
        # Lauer 5
        out = self.conv5(out)
        out = self.relu5(out)
        # Lauer 6
        out = self.conv6(out)
        out = self.relu6(out)
        # Lauer 7
        out = self.conv7(out)
        out = self.relu7(out)
        # Lauer 8
        out = self.maxpool8(out)

        out = out.view(out.size(0), -1)
        # Lauer 9
        out = self.dropout1(out)
        out = self.linear1(out)
        out = self.relu9(out)
        # Lauer 10
        out = self.dropout2(out)
        out = self.linear2(out)
        out = self.relu10(out)
        # Lauer 11
        out = self.linear3(out)
        return out
