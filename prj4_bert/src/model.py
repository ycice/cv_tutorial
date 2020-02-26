from torch import nn
import torch.nn.functional as F


class BertFc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 4)

    def forward(self, x):
        out = F.relu(self.fc(x))
        return out
