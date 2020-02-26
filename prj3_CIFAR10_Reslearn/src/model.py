import torch
from torch import nn
import torch.nn.functional as F


class RLBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RLBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        if in_channel != out_channel * 4:
            self.shortcut = nn.Conv2d(in_channel, out_channel * 4, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x: torch.FloatTensor):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        if self.in_channel != self.out_channel * 4:
            out = self.conv3(out) + self.shortcut(x)
        else:
            out = self.conv3(out) + x
        out = F.relu(self.bn3(out))
        return out


class ResidualLearning(nn.Module):
    def __init__(self, floors: list, channels: list):
        super(ResidualLearning, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)
        self.layer1 = self.make_layer(floors[0], 64, channels[0])
        self.layer2 = self.make_layer(floors[1], channels[0] * 4, channels[1])
        self.pol = nn.Conv2d(channels[1] * 4, channels[1] * 4, kernel_size=3, stride=2, padding=1)
        self.layer3 = self.make_layer(floors[2], channels[1] * 4, channels[2])
        self.layer4 = self.make_layer(floors[3], channels[2] * 4, channels[3])
        self.fc1 = nn.Linear(channels[3] * 4 * 8 * 8, 10)

    def make_layer(self, deep: int, in_channel, out_channel):
        layers = []
        for i in range(deep):
            layers.append(RLBlock(in_channel, out_channel))
            in_channel = out_channel * 4
        return nn.Sequential(*layers)  # *을 붙이면 list를 개별 원소들로 바꿔줌
        # **kwarg는 [a=1, b=2, c=3]을 딕셔너리처럼 동작하게 해줌

    def forward(self, x: torch.FloatTensor):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pol(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view((out.shape[0], -1))
        out = self.fc1(out)
        return out


if __name__ == '__main__':
    random_img = torch.randn((16, 3, 32, 32))
    model = ResidualLearning([3, 4, 6, 3], [64, 128, 256, 512])
    random_y = model(random_img)
