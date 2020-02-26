from torch import nn


class BertFc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 4)

    def forward(self, x):
        out = self.fc(x)
        return out
