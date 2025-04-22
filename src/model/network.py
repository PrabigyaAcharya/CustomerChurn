import torch.nn as nn

class ChurnNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=19, out_features = 24)
        self.layer_2 = nn.Linear(in_features=24, out_features=1)
    
    def forward(self, x):
        return self.layer_2(self.layer_1(x))