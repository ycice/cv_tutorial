from torch import nn


class BertSnliFc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 3)

    def forward(self, x):
        out = self.fc(x)
        return out
