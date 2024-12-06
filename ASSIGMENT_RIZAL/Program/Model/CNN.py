import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_dim=32, input_c=3, output=6, device='cuda'):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), 
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.fc = nn.Linear(in_features=50 * 5 * 5, out_features=output)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x