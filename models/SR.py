import torch
import torch.nn as nn

class SRNet(nn.Module):
    def __init__(self, padding=True, num_channels=3, in_len=2000, out_len1=10000, out_len2=20000):
        super(SRNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=9, padding=4*int(padding), padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=1, padding=0*int(padding), padding_mode='replicate'),  # n1 * 1 * 1 * n2
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 3, kernel_size=5, padding=2*int(padding), padding_mode='replicate'),
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=9, padding=4*int(padding), padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=1, padding=0*int(padding), padding_mode='replicate'),  # n1 * 1 * 1 * n2
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 3, kernel_size=5, padding=2*int(padding), padding_mode='replicate'),
            )


        self.fc1 = nn.Linear(in_len, out_len1)
        self.fc2 = nn.Linear(out_len1, out_len2)
  
        
        self.tail1 = nn.Conv1d(3, num_channels, kernel_size=27, padding=13*int(padding))
        self.tail2 = nn.Conv1d(3, num_channels, kernel_size=27, padding=13*int(padding))

        self.apply(weight_init)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        out1 = self.tail1(x)

        x = self.conv2(out1)
        x = self.fc2(x)
        out2 = self.tail2(x)
        return out1, out2


    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)