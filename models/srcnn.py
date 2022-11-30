import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, padding=True, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=1001, padding=500*int(padding), padding_mode='replicate'),
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=201, padding=100*int(padding), padding_mode='replicate'),  # n1 * 1 * 1 * n2
            nn.ReLU(inplace=True),
            )
        self.conv3 = nn.Conv1d(32, num_channels, kernel_size=501, padding=250*int(padding), padding_mode='replicate')
        self.init_weights()
    
    def forward(self, x):
        # res = x
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        return x
    
    def init_weights(self):
        for L in self.conv1:
            if isinstance(L, nn.Conv1d):
                L.weight.data.normal_(mean=0.0, std=0.001)
                L.bias.data.zero_()
        for L in self.conv2:
            if isinstance(L, nn.Conv1d):
                L.weight.data.normal_(mean=0.0, std=0.001)
                L.bias.data.zero_()
        self.conv3.weight.data.normal_(mean=0.0, std=0.001)
        self.conv3.bias.data.zero_()