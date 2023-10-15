import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channel, n_filters):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channel, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters * 4, n_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters * 8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)